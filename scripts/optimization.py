#!/usr/bin/env python3
from fileinput import filename

import rospy
from kuka_humanoids.motion_interface import MoveGroupPythonInterface
from iiwa_msgs.msg import JointTorque

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
import numpy as np
import roboticstoolbox as rbt
from roboticstoolbox import ERobot
from roboticstoolbox import Link, ET
from scipy.optimize import minimize, direct
import time
import os
import csv
from spatialmath import SE3,SO3
from scipy.signal import butter, lfilter, lfilter_zi
import matplotlib.pyplot as plt

def loginfo_green(message):
    """Prints a message to the ROS log in green color.

    Args:
        message (str): The message to be printed.
    """
    green_start = '\033[92m'
    color_reset = '\033[0m'
    rospy.loginfo(green_start + str(message) + color_reset)

class CollisionDetection:
    """Initializes the CollisionDetection class.

    Args:
        thresholds (list[float]): A list of torque thresholds for each joint.
        buffer_size (int, optional): The size of the torque buffer.
    """
    def __init__(self, thresholds, buffer_size = 50, filter_cutoff_hz=10.0, filter_sampling_hz=356.0, filter_order=2):
        rospy.init_node("examples_node")  # Initialize the ROS node

        # Initialize motion interface for robot
        self.move_group = MoveGroupPythonInterface("iiwa_arm")

        # Force threshold for each one of the joints
        self.torque_thresholds = thresholds

        # Initialize external forces (torques)
        self.current_external_torques = [0.0] * 7

        # Initialize robot model from robotics_toolbox using our URDF
        self.robot = ERobot.URDF("/home/docker/kuka_ws/src/iiwa_stack/iiwa_description/urdf/iiwa7.urdf.xacro") # Use roboticstoolbox's KUKA LWR model
        # Initialize torque buffer
        self.torque_buffer = []

        # Initialize size of torque buffer
        self.buffer_size = buffer_size
        self.frame_name = 0

        # --- Low-pass Filter Setup ---
        self.filter_order = filter_order
        self.filter_sampling_hz = filter_sampling_hz  # IMPORTANT: Adjust this to the actual publishing rate of /iiwa/state/ExternalJointTorque
        self.filter_cutoff_hz = filter_cutoff_hz
        nyquist = 0.5 * self.filter_sampling_hz
        normalized_cutoff = self.filter_cutoff_hz / nyquist
        # Design the Butterworth filter
        self.filter_b, self.filter_a = butter(self.filter_order, normalized_cutoff, btype='low', analog=False)
        # Initialize filter state (zi) for each of the 7 joints
        # We need separate states because we filter each joint independently
        zi_initial = lfilter_zi(self.filter_b, self.filter_a)
        self.filter_zi = [np.copy(zi_initial) for _ in range(7)]  # Initialize state for each joint
        rospy.loginfo(
            f"Low-pass filter initialized: Order={self.filter_order}, Cutoff={self.filter_cutoff_hz} Hz, Sampling={self.filter_sampling_hz} Hz")
        rospy.loginfo(f"Filter coefficients (b): {self.filter_b}")
        rospy.loginfo(f"Filter coefficients (a): {self.filter_a}")
        # -----------------------------

        # Link lengths approximation (from URDF visual inspection and some educated guesses)
        self.link_lengths = [
            0.15, # link 0
            0.19, # link 1
            0.21, # link 2
            0.19, # link 3
            0.21, # link 4
            0.19, # link 5
            0.0607, # link 6
            0.0623 # link 7
        ]
        self.total_length = sum(self.link_lengths)

        # Keep track of added collision objects
        self.added_collision_objects = []
        self.collision_object_counter = 0

        # Subscribe to the external joint torque topic
        rospy.Subscriber("/iiwa/state/ExternalJointTorque", JointTorque, callback=self.torque_callback)

        # Publisher for RViz
        self.marker_pub = rospy.Publisher("/iiwa/visualization_marker", Marker, queue_size=10)

        self.filtered_torque_pub = rospy.Publisher("/iiwa/state/FilteredExternalJointTorque", JointTorque, queue_size=10)
        rospy.loginfo(f"Publishing filtered torques on: {self.filtered_torque_pub.name}")

        # Counter for unique marker IDs
        self.collision_marker_id_counter = 0
        self.published_marker_ids = []

        self.collision_marker_id_counter = 0
        self.added_collision_object_names = []

        self.direct_evaluations = []
        self.plot_counter = 0  # To save plots with unique names

        self.output_csv_filename = "contact_points_log.csv"
        self.contact_point_log_counter = 0

    def objective_function_wrapper(self, x, q, tau_f_measured):
        """
        Wrapper for the objective function to log evaluations for visualization.

        Args:
            x: Optimization variables [s, phi].
            q: Current joint angles.
            tau_f_measured: Measured external joint torques.

        Returns:
            objective_val: Scalar objective function value.
        """
        # Call the original objective function
        objective_val = self.objective_function_kuka(x, q, tau_f_measured)

        # Store the evaluation point (s, phi) and the result
        if self.direct_evaluations is not None:  # Check if logging is enabled
            # Ensure x has at least 2 elements before accessing
            if len(x) >= 2:
                self.direct_evaluations.append((x[0], x[1], objective_val))
            else:
                rospy.logwarn_throttle(5, f"Objective wrapper received x with unexpected shape: {x}")

        return objective_val

    def torque_callback(self, msg):
        """Callback function to update the current external joint torques.

        Args:
            msg (iiwa_msgs.msg.JointTorque): The received ExternalJointTorque message.
        """

        raw_torques = [
            msg.torque.a1, msg.torque.a2, msg.torque.a3,
            msg.torque.a4, msg.torque.a5, msg.torque.a6, msg.torque.a7
        ]
        self.current_external_torques = raw_torques
        self.torque_buffer.append(raw_torques)

        filtered_torques = [0.0] * 7
        for i in range(7):
            filtered_val, self.filter_zi[i] = lfilter(self.filter_b,
                                                      self.filter_a,
                                                      [raw_torques[i]],
                                                      zi=self.filter_zi[i])
            filtered_torques[i] = filtered_val[0]

        self.torque_buffer.append(filtered_torques)

        # Limit buffer to buffer size
        if len(self.torque_buffer) > self.buffer_size:
            self.torque_buffer.pop(0)

        self.current_external_torques = filtered_torques

        filtered_torque_msg = JointTorque()
        filtered_torque_msg.header = msg.header

        filtered_torque_msg.torque.a1 = filtered_torques[0]
        filtered_torque_msg.torque.a2 = filtered_torques[1]
        filtered_torque_msg.torque.a3 = filtered_torques[2]
        filtered_torque_msg.torque.a4 = filtered_torques[3]
        filtered_torque_msg.torque.a5 = filtered_torques[4]
        filtered_torque_msg.torque.a6 = filtered_torques[5]
        filtered_torque_msg.torque.a7 = filtered_torques[6]

        # Publish the message
        self.filtered_torque_pub.publish(filtered_torque_msg)

    def publish_collision_point(self, collided_link, point,force_vector, marker_id=1, frame_id_override=None, fault_isolation = False):
        """Publishes a sphere marker to RViz representing a collision point.

        Args:
            collided_link (int): The index of the link in collision.
            point (list[float]): The coordinates of the collision point [x, y, z].
            force_vector (numpy.ndarray): The estimated force vector at the collision point.
            marker_id (int, optional): The ID of the marker.
            frame_id_override (str, optional): An optional frame ID to override the default.
        """

        frameid = {1: "iiwa_link_1",
                   2: "iiwa_link_2",
                   3: "iiwa_link_3",
                   4: "iiwa_link_4",
                   5: "iiwa_link_5",
                   6: "iiwa_link_6",
                   7: "iiwa_link_7"}
        # Use frame_id_override if provided, otherwise use the default frameid mapping
        marker_frame_id = frame_id_override if frame_id_override else frameid[collided_link]
        # Create a Marker object for visualizing the contact point
        marker = Marker()
        marker.header.frame_id = marker_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_point"
        marker.id = marker_id
        marker.type = Marker.SPHERE  # Use a sphere to represent a single point
        marker.action = Marker.ADD

        marker.scale.x = 0.16  # Sphere radius
        marker.scale.y = 0.16
        marker.scale.z = 0.16

        # not a faulty isolation
        if not fault_isolation:
            marker.color.r = 255/255  # Red color
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.65

        if fault_isolation:
            marker.color.r = 255/255 # Orange color
            marker.color.g = 128/255
            marker.color.b = 0.0
            marker.color.a = 0.25

        # Shift the point origin in the negative direction of the force vector by 0.08
        force_norm = np.linalg.norm(force_vector)
        if force_norm > 1e-6:  # Avoid division by zero
            direction = force_vector / force_norm
            shift = -0.12 * direction
            shifted_point = np.array(point) + shift
            marker.pose.position = Point(*point)  # The shifted point
        else:
            marker.pose.position = Point(*point)  #
            rospy.logwarn("Estimated force vector is close to zero, not shifting collision point.")
        # No rotation quaternion
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        # Publish the marker
        self.marker_pub.publish(marker)

        #collision_object_name = f"collision_sphere_{marker_id}"
        collision_object_name = f"collision_sphere"
        # Add the same sphere as a collision object to MoveIt
        """        self.create_sphere_bbox(
            MoveGroupArm=self.move_group,
            center=shifted_point,
            radius=marker.scale.x / 4.0, # Use radius (scale is diameter)
            name=collision_object_name
        )"""
        #self.added_collision_object_names.append(collision_object_name)
        #time.sleep(8)
        #self.remove_sphere_bbox(self.move_group)
        return shifted_point

    def publish_collision_point2(self, collided_link, point, force_vector, marker_id=1, frame_id_override=None):
        """Publishes a sphere marker to RViz representing a collision point.

        Args:
            collided_link (int): The index of the link in collision.
            point (list[float]): The coordinates of the collision point [x, y, z].
            marker_id (int, optional): The ID of the marker.
            frame_id_override (str, optional): An optional frame ID to override the default.
        """

        frameid = {1: "iiwa_link_1",
                   2: "iiwa_link_2",
                   3: "iiwa_link_3",
                   4: "iiwa_link_4",
                   5: "iiwa_link_5",
                   6: "iiwa_link_6",
                   7: "iiwa_link_7"}
        # Use frame_id_override if provided, otherwise use the default frameid mapping
        marker_frame_id = frame_id_override if frame_id_override else frameid[collided_link]
        # Create a Marker object for visualizing the contact point
        marker = Marker()
        marker.header.frame_id = marker_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_object"
        marker.id = marker_id
        marker.type = Marker.SPHERE  # Use a sphere to represent a single point
        marker.action = Marker.ADD

        marker.scale.x = 0.16  # Sphere radius
        marker.scale.y = 0.16
        marker.scale.z = 0.16

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 255/255
        marker.color.a = 0.3

        coll_point = force_vector.copy()
        coll_point[2]  = point[2]

        self.contact_point_log_counter += 1
        self.log_contact_point_to_csv(self.contact_point_log_counter,coll_point)
        marker.pose.position = Point(*coll_point)  # The shifted point
        # No rotation quaternion
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        # Publish the marker
        self.marker_pub.publish(marker)

    def log_contact_point_to_csv(self, point_number, contact_point_coords):
        """
        Writes the contact point number and its coordinates (x, y, z) into a CSV file.

        Args:
            point_number (int): The sequential number of the contact point.
            contact_point_coords (numpy.ndarray or list/tuple):
                The coordinates [x, y, z] of the contact point.
        """
        filename = self.output_csv_filename
        # Check if file exists AND is not empty to determine if header is needed
        file_exists_and_not_empty = os.path.isfile(filename) and os.path.getsize(filename) > 0

        with open(filename, 'a', newline='') as csvfile:  # 'a' for append mode
            csv_writer = csv.writer(csvfile)

            if not file_exists_and_not_empty:
                # Write header if file is new or empty
                csv_writer.writerow(["point_number", "x", "y", "z"])

            coords_list = contact_point_coords

            csv_writer.writerow([point_number, coords_list[0], coords_list[1], coords_list[2]])
            rospy.loginfo(
                f"Logged contact point #{point_number} to {filename}: [{coords_list[0]:.4f}, {coords_list[1]:.4f}, {coords_list[2]:.4f}]")

    def publish_collision_point_eef(self, collided_link, point, force_vector, marker_id=1, frame_id_override=None):
        """Publishes a sphere marker to RViz representing a collision point.

        Args:
            collided_link (int): The index of the link in collision.
            point (list[float]): The coordinates of the collision point [x, y, z].
            marker_id (int, optional): The ID of the marker.
            frame_id_override (str, optional): An optional frame ID to override the default.
        """

        frameid = {1: "iiwa_link_1",
                   2: "iiwa_link_2",
                   3: "iiwa_link_3",
                   4: "iiwa_link_4",
                   5: "iiwa_link_5",
                   6: "iiwa_link_6",
                   7: "iiwa_link_7"}
        # Use frame_id_override if provided, otherwise use the default frameid mapping
        marker_frame_id = frame_id_override if frame_id_override else frameid[collided_link]
        # Create a Marker object for visualizing the contact point
        marker = Marker()
        marker.header.frame_id = marker_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_object"
        marker.id = marker_id
        marker.type = Marker.SPHERE  # Use a sphere to represent a single point
        marker.action = Marker.ADD

        marker.scale.x = 0.16  # Sphere radius
        marker.scale.y = 0.16
        marker.scale.z = 0.16

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 255/255
        marker.color.a = 0.3

        coll_point = force_vector.copy()
        coll_point[0]  = point[0]
        coll_point[2] = point[2]

        marker.pose.position = Point(*coll_point)  # The shifted point
        # No rotation quaternion
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        self.publish_collision_point_center(coll_point,marker_id)

        # Publish the marker
        self.marker_pub.publish(marker)

    def publish_collision_point_center(self, point,marker_id=1):
        """Publishes a sphere marker to RViz representing a collision point.

        Args:
            collided_link (int): The index of the link in collision.
            point (list[float]): The coordinates of the collision point [x, y, z].
            marker_id (int, optional): The ID of the marker.
            frame_id_override (str, optional): An optional frame ID to override the default.
        """

        # Use frame_id_override if provided, otherwise use the default frameid mapping
        marker_frame_id = "world"
        # Create a Marker object for visualizing the contact point
        marker = Marker()
        marker.header.frame_id = marker_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_object_center"
        marker.id = marker_id
        marker.type = Marker.SPHERE  # Use a sphere to represent a single point
        marker.action = Marker.ADD

        marker.scale.x = 0.06  # Sphere radius
        marker.scale.y = 0.06
        marker.scale.z = 0.06

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.65


        marker.pose.position = Point(*point)  # The shifted point
        # No rotation quaternion
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        # Publish the marker
        self.marker_pub.publish(marker)

    def clear_all_collision_objects(self):
        """Removes all tracked collision objects from the MoveIt planning scene."""
        rospy.loginfo(f"Removing {len(self.added_collision_object_names)} collision objects from MoveIt...")
        scene = self.move_group.planning_scene # Get PlanningSceneInterface instance

        for name in self.added_collision_object_names:
            rospy.logdebug(f"Removing collision object: {name}")
            scene.removeCollisionObject(name)
            # Short sleep might be needed for scene updates, test if necessary
            # rospy.sleep(0.1)

        rospy.loginfo("Finished removing collision objects.")
        # Clear the list of tracked names
        self.added_collision_object_names = []
        
    def remove_sphere_bbox(self,MoveGroupArm, name="collision_sphere_1"):
        """Removes a sphere collision object from the MoveIt planning scene.

        Args:
            MoveGroupArm (MoveGroupPythonInterface): The MoveGroupPythonInterface object.
            name (str, optional): The name of the collision object to remove.
        """
        MoveGroupArm.planning_scene.removeCollisionObject(name)
        #rospy.loginfo(f"Removed collision object '{name}' from MoveIt.")

    def get_n_valid_torques(self,n):
        """Retrieves the n-th most recent set of torques from the buffer.

        Args:
            n (int): The index of the torque set to retrieve (0 for the latest).

        Returns:
            list[float]: A list of 7 torque values, or a list of zeros if not enough data is available.
        """
        if self.torque_buffer and len(self.torque_buffer) >= n:
            return self.torque_buffer[n]

        # Default value
        return [0.0] * 7

    def isolate_collision_link(self, collision_torques):
        """Identifies the link that is most likely in collision based on torque readings.

        Args:
            collision_torques (list[float]): The external torques measured at the moment of collision.

        Returns:
            tuple[list[float], int]: A tuple containing the (potentially masked) collision torques and the index of the collided link (1-7).
        """
        collided_link = None

        # Lower index is older value
        collision_offset = self.get_n_valid_torques(0) # half a second

        # These are the torques from the collision when we exceed the threshold
        #print(f"COLLISION TORQUES: {collision_torques}")
        # Thees are the pre-collision torques
        #print(f"COLLISION OFFSET: {collision_offset}")

        # Calculate the difference between the collision torques and the collision offset
        torque_difference = np.subtract(collision_torques, collision_offset)

        # Threshold to zero residuals
        torque_threshold = 0.03

        # Apply the threshold to make small residuals zero
        collision_torques_masked = np.where(np.abs(torque_difference) < torque_threshold, 0.0, torque_difference)

        num_joints = len(collision_torques_masked)

        # Iterate from the last joint index (num_joints - 1) down to 0
        for i in range(num_joints - 1, -1, -1):
            torque = collision_torques_masked[i]
            if abs(torque) > 0.0:
                collided_link = i + 1
                break

        if collided_link is not None:
            loginfo_green(f"Collided link index: {collided_link}")
        else:
            rospy.logerr("Collided link was not found")
            collided_link = 4

        #return collision_torques, collided_link
        #print(f"torque difference: {collision_torques_masked}")
        return collision_torques_masked, collided_link

    def get_link_jacobian_and_point(self, joint_angles, s, plotting = False):
        """Calculates the Jacobian and a point on the robot link for a given normalized position.

        Args:
            joint_angles (list[float]): The current joint angles of the robot.
            s (float): A normalized position along the robot's total length (between 0 and 1).
            plotting (bool, optional): If True, plots the robot configuration.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, int]: A tuple containing the Jacobian matrix (6x7), the contact point coordinates in the base frame (3x1), and the index of the collided link (1-7).
        """

        q = joint_angles
        cumulative_lengths = np.cumsum(self.link_lengths) # [0.15 0.34 0.55 0.74 0.95 1.14 1.2007 1.2457]
        s_total_length = s * self.total_length
        collided_link_index = None
        a_i = 0.0 # Distance from joint i to point po

        if s_total_length <= cumulative_lengths[1]: # Contact on link 1
            collided_link_index = 1
            a_i = s_total_length
        elif s_total_length <= cumulative_lengths[2]: # Contact on link 2
            collided_link_index = 2
            a_i = s_total_length - cumulative_lengths[1]
        elif s_total_length <= cumulative_lengths[3]: # Contact on link 3
            collided_link_index = 3
            a_i = s_total_length - cumulative_lengths[2]
        elif s_total_length <= cumulative_lengths[4]: # Contact on link 4
            collided_link_index = 4
            a_i = s_total_length - cumulative_lengths[3]
        elif s_total_length <= cumulative_lengths[5]: # Contact on link 5
            collided_link_index = 5
            a_i = s_total_length - cumulative_lengths[4]
        elif s_total_length <= cumulative_lengths[6]: # Contact on link 6
            collided_link_index = 6
            a_i = s_total_length - cumulative_lengths[5]
        elif s_total_length <= cumulative_lengths[7]: # Contact on link 7
            collided_link_index = 7
            a_i = s_total_length - cumulative_lengths[6]
        else:
            collided_link_index = 7
            a_i = self.link_lengths[7]
            print(f"a_i = {a_i}")
            rospy.logwarn("s value out of expected range, defaulting to link 7 end.")

        link_name = f"iiwa_link_{collided_link_index}"

        # Virtual structure for collision point calculation
        iiwa_ets_list = []
        iiwa_ets_list.append(ET.tz(0.15) * ET.Rz(jindex=0))  # link 1
        iiwa_ets_list.append(ET.tz(0.19) * ET.Ry(jindex=1))  # link 2
        iiwa_ets_list.append(ET.tz(0.21) * ET.Rz(jindex=2))  # link 3
        iiwa_ets_list.append(ET.tz(0.19) * ET.Ry(jindex=3, flip=True))  # link 4
        iiwa_ets_list.append(ET.tz(0.21) * ET.Rz(jindex=4))  # link 5
        iiwa_ets_list.append(ET.tz(0.19) * ET.Ry(jindex=5))  # link 6
        iiwa_ets_list.append(ET.tz(0.0607) * ET.Rz(jindex=6))  # link 7
        # Create a new robot structure dynamically using ETS
        new_robot_ets = ET.tz(0)
        for i, iiwa_ets in enumerate(iiwa_ets_list):
            if i>= collided_link_index:
                break
            for ets in iiwa_ets:
                new_robot_ets *= ets
        # Add virtual link for collision point using ETS
        collision_point_et = ET.tz(a_i)
        new_robot_ets *= collision_point_et
        coords =q[:collided_link_index]
        plot_filename = f'jacobian_search_{self.frame_name}.png'
        if plotting:
            plt.figure()
            new_robot_ets.plot(q[:collided_link_index],block = False, backend = 'pyplot')
            plt.savefig(plot_filename)
            plt.close()

        # Calculate FKine and Jacobian using the new robot model
        T_link_base = new_robot_ets.fkine(coords)  # FKine to the end of the ETS chain (collision point)
        po_base_frame = T_link_base.t
        T_link_SE3 = SE3(T_link_base)

        jacobian_po_base = new_robot_ets.jacob0(coords)  # Jacobian to the end of the ETS chain

        self.frame_name += 1

        return jacobian_po_base, po_base_frame, collided_link_index, T_link_SE3

    def objective_function_kuka(self, x, q, tau_f_measured):
        """
        Objective function to minimize for KUKA robot contact force estimation.

        Args:
            x: Optimization variables [s, phi] where s is normalized position (0 to 1)
               and phi is the angle of the force in the link frame (0 to 2*pi).
            q: Current joint angles (7x1 numpy array).
            tau_f_measured: Measured external joint torques (7x1 numpy array).

        Returns:
            objective_val: Scalar objective function value.
        """
        s_val = x[0]
        phi = x[1]
        f_bar_s = np.array([np.cos(phi), np.sin(phi), 0])
        jacobian_po_s, _, collided_link,_= self.get_link_jacobian_and_point(q, s_val)

        # Reduce Jacobian columns based on collided link (Eq. 8 in paper)
        jacobian_po_s_reduced = jacobian_po_s[:, :collided_link]
        zero_cols = np.zeros((6, 7 - collided_link))
        jacobian_po_s_contact = np.concatenate((jacobian_po_s_reduced, zero_cols), axis=1)

        # Use only positional Jacobian part (3x7)
        tau_bar_s = jacobian_po_s_contact[:3, :].T @ f_bar_s
        tau_bar_s_norm = np.linalg.norm(tau_bar_s)
        if tau_bar_s_norm < 1e-9:  # Handle near-zero norm
            return 1e6

        tau_s_estimated = np.linalg.norm(tau_f_measured) * (tau_bar_s/tau_bar_s_norm) #(Eq. 25 in paper)

        tau_f_measured_norm = tau_f_measured / np.linalg.norm(tau_f_measured)
        tau_s_estimated_norm = tau_s_estimated / np.linalg.norm(tau_s_estimated) # From the older version of the paper
        def count_same_sign(arr1, arr2):

            sign_arr1 = np.sign(arr1)
            sign_arr2 = np.sign(arr2)

            same_sign_mask = (sign_arr1 == sign_arr2)

            cnt = np.sum(same_sign_mask)

            return cnt

        #objective_val = np.linalg.norm(tau_f_measured_norm - tau_s_estimated_norm) + count_same_sign(tau_f_measured, tau_bar_s_norm)/7
        objective_val = np.linalg.norm(tau_f_measured_norm - tau_s_estimated_norm)
        return objective_val

    def initial_condition_selection(self, collided_link):
        """Calculates the initial value for the s into optimization procces.

        Args:
            collided_link: Link index to calculate initial value for.
        Returns:
            initial_s (float) : Initial value for the s optimization.
        """
        cumulative_lengths = np.cumsum(self.link_lengths)  # [0.15 0.34 0.55 0.74 0.95 1.14 1.2007 1.2457]
        link_center = (cumulative_lengths[collided_link] - cumulative_lengths[collided_link - 1]) / 2
        s_initial = (cumulative_lengths[collided_link] - link_center) / self.total_length
        return s_initial

    def isolate_collision_point_optimization(self, collided_link_initial, initial_s, joint_angles, collision_torques):
        """Estimates the collision point and force using a two-stage optimization process.

        Args:
            initial_s (float) : Initial value for the s optimization.
            joint_angles (list[float]): The current joint angles of the robot.
            collision_torques (list[float]): The measured external joint torques.

        Returns:
            tuple[int, numpy.ndarray, numpy.ndarray]: A tuple containing the index of the collided link (1-7), the estimated contact point in the base frame (3x1), and the estimated contact force in the link frame (3x1). Returns (None, None, None) if optimization fails.
        """

        q = joint_angles
        tau_f_measured = np.array(collision_torques)
        if collided_link_initial is not None:
            cumulative_lengths = np.cumsum(self.link_lengths)
            s_min_link = cumulative_lengths[
                             collided_link_initial - 1] / self.total_length if collided_link_initial > 1 else 0.0
            s_max_link = cumulative_lengths[collided_link_initial] / self.total_length
            s_min_link = max(0.0, s_min_link)
            s_max_link = min(1.0, s_max_link)
            print(s_min_link, s_max_link)

            #bounds = [(s_min_link, s_max_link), (0.0, 2 * np.pi)]  # Use these bounds for optimization

        # Define bounds for s (normalized position along the robot) and phi
        bounds = [(0.0, 1.0), (0.0, 2 * np.pi)]  # s is between 0 and 1 and phi is 0 and 2pi

        # Initial guess for s and phi
        initial_guess = [initial_s, 0.0]

        # --- Global Optimization (DIRECT) --- solid parameters 26.03.2025
        print("\n" + "=" * 60)
        print("Starting global optimization (DIRECT)...")
        print("=" * 60)
        start_time_global = time.time()
        global_optimization_result = direct(
            self.objective_function_kuka,
            bounds,
            args=(q, tau_f_measured),
            eps=0.01,
            vol_tol=0.001,
            locally_biased=False
        )
        end_time_global = time.time()
        duration_global = end_time_global - start_time_global
        print(f"Global optimization took: {duration_global:.6f} seconds")

        if hasattr(global_optimization_result, 'success') and global_optimization_result.success:
            best_global_x = global_optimization_result.x
            print(
                f"Global optimization 2 successful. Best [s, phi]: {best_global_x}, Objective value: {global_optimization_result.fun}")
            initial_guess = best_global_x  # Use the global optimum as the starting point for local optimization
        else:
            rospy.logwarn(
                f"Global optimization failed: {getattr(global_optimization_result, 'message', 'No message')}, using initial guess for local optimization.")
        print("-" * 60)
        # --- Local Optimization (SLSQP) ---
        print("\n" + "=" * 60)
        print("Starting local optimization (SLSQP)...")
        print("=" * 60)
        start_time_local = time.time()  # Record start time
        optimization_result = minimize(
            self.objective_function_kuka,
            initial_guess,
            args=(q, tau_f_measured),
            bounds=bounds,
            method='SLSQP'
        )
        end_time_local = time.time()
        duration_local = end_time_local - start_time_local
        print(f"Local SLSQP optimization took: {duration_local:.6f} seconds")

        if hasattr(optimization_result, 'success') and optimization_result.success:
            best_s_phi = optimization_result.x
            best_s = best_s_phi[0]
            best_phi = best_s_phi[1]
            min_objective_value = optimization_result.fun
            print(
                f"Local optimization 2 successful. Best [s, phi]: {best_s_phi}, Objective value: {min_objective_value}")

            jacobian_po_best_s, po_best_s, collided_link, T_link_SE3 = self.get_link_jacobian_and_point(q, best_s)
            if jacobian_po_best_s is None:
                rospy.logwarn("Could not get Jacobian for the best s.")
                return None, None, None, None, None
            print("-"*60)

            f_bar_s_estimated_collision_frame = np.array([np.cos(best_phi), np.sin(best_phi), 0])

            jacobian_po_s_reduced_world = jacobian_po_best_s[:, :collided_link]
            zero_cols_world = np.zeros((6, 7 - collided_link))
            jacobian_po_s_contact_world = np.concatenate((jacobian_po_s_reduced_world, zero_cols_world), axis=1)

            rotation_matrix_world = T_link_SE3.R

            force_direction_world = rotation_matrix_world @ f_bar_s_estimated_collision_frame

            tau_bar_s_world = jacobian_po_s_contact_world[:3, :].T @ force_direction_world
            tau_bar_s_world_norm = np.linalg.norm(tau_bar_s_world)

            f_s_magnitude_estimated = 0.0
            force_vector_world = np.zeros(3)

            if tau_bar_s_world_norm > 1e-9:
                f_s_magnitude_estimated = np.linalg.norm(tau_s_estimated) / tau_bar_s_world_norm
                force_vector_world = f_s_magnitude_estimated * force_direction_world
                tau_s_estimated = jacobian_po_s_contact_world[:3, :].T @ force_vector_world

            # --- Print Results ---
            label_width = 35
            print("\n" + "=" * 60)
            print("Robot Collision and Force Estimation Details")
            print("=" * 60)
            print(
                f"{'Est. Force Direction (Collision Frame):':<{label_width}} {f_bar_s_estimated_collision_frame} (x,y,z) [Unit Vector]")
            print(
                f"{'Est. Force Vector (World Frame):':<{label_width}} {force_vector_world} (x,y,z) [N]")  # Print world vector
            print(f"{'Est. Contact Force Magnitude:':<{label_width}} {f_s_magnitude_estimated:.3f} [N]")
            print(f"{'Est. Contact Point (Base Frame):':<{label_width}} {po_best_s} (x,y,z) [m]")
            print(f"{'Measured Torques:':<{label_width}} {tau_f_measured} [Nm]")
            print(f"{'Estimated Torques:':<{label_width}} {tau_s_estimated} [Nm]")
            print("-" * 60)
            print(f"{'Calculated Colliding Link:':<{label_width}} {collided_link}")
            print("=" * 60 + "\n")

            return collided_link, po_best_s, force_vector_world, best_s, f_s_magnitude_estimated

        else:
            rospy.logwarn(f"Local optimization failed: {getattr(optimization_result, 'message', 'No message')}")
            # Return None for the force vector too
            return None, None, None, None, None, None

    def publish_arrow_world(self, start_point, end_point, frame_id, marker_id):
        """Publishes an arrow marker between two points in a specified frame."""
        marker = Marker()
        marker.header.frame_id = frame_id  # e.g., "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_vector_world"  # Use a distinct namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow appearance
        marker.scale.x = 0.02  # Shaft diameter
        marker.scale.y = 0.04  # Head diameter
        marker.scale.z = 0.04  # Head length

        marker.color.r = 0.0  # Green color
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Define arrow start and end points directly
        marker.points = [Point(*start_point), Point(*end_point)]

        # Use identity pose as points define the arrow's position/orientation
        marker.pose.position = Point(0, 0, 0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        self.marker_pub.publish(marker)
        rospy.logdebug(
            f"Published world frame arrow marker {marker_id} from {start_point} to {end_point} in frame '{frame_id}'")

    def publish_arrow_oposite_world(self, start_point, end_point, frame_id, marker_id):
        """Publishes an arrow marker between two points in a specified frame."""
        marker = Marker()
        marker.header.frame_id = frame_id  # e.g., "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_vector_opposite_world"  # Use a distinct namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow appearance
        marker.scale.x = 0.02  # Shaft diameter
        marker.scale.y = 0.04  # Head diameter
        marker.scale.z = 0.04  # Head length

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.7

        # Define arrow start and end points directly
        marker.points = [Point(*start_point), Point(*end_point)]

        # Use identity pose as points define the arrow's position/orientation
        marker.pose.position = Point(0, 0, 0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        self.marker_pub.publish(marker)
        rospy.logdebug(
            f"Published world frame arrow marker {marker_id} from {start_point} to {end_point} in frame '{frame_id}'")

    def visualize_contact_estimation(self, collided_link, contact_point_world, force_vector_world,fault_isolation = False):
        """Publishes RViz markers to visualize the estimated contact point (world) and force vector (world).

        Args:
            collided_link (int): The index of the link in collision (1-7).
            contact_point_world (numpy.ndarray): The estimated contact point coordinates in the world frame (3x1).
            force_vector_world (numpy.ndarray): The estimated contact force vector in the world frame (3x1).
        """
        self.collision_marker_id_counter += 1
        # Check if inputs are valid before proceeding
        if collided_link is not None and contact_point_world is not None and force_vector_world is not None and force_vector_world.size == 3:

            shifted_point = self.publish_collision_point(collided_link, contact_point_world, force_vector_world,
                                                         frame_id_override="world",
                                                         marker_id=self.collision_marker_id_counter,
                                                         fault_isolation = fault_isolation)

            arrow_visual_length = 0.25  # meters

            force_magnitude = np.linalg.norm(force_vector_world)

            if force_magnitude > 1e-6:
                force_direction_world = force_vector_world / force_magnitude
                end_point_vis_world = contact_point_world + force_direction_world * arrow_visual_length
            else:
                rospy.logwarn("Estimated force magnitude is near zero. Drawing small default arrow.")
                force_direction_world = np.array([0.0, 0.0, 1.0])
                end_point_vis_world = contact_point_world + force_direction_world * 0.05

            self.publish_arrow_world(start_point=contact_point_world,
                                     end_point=end_point_vis_world,
                                     frame_id="world",
                                     marker_id=self.collision_marker_id_counter)

            opposite_arrow_legth = -0.25
            if force_magnitude > 1e-6:
                force_direction_world = force_vector_world / force_magnitude
                end_point_vis_opp_world = contact_point_world + force_direction_world * opposite_arrow_legth
                end_point_vis_opp_world[2] = contact_point_world[2]
            else:
                rospy.logwarn("Estimated force magnitude is near zero. Drawing small default arrow.")
                force_direction_world = np.array([0.0, 0.0, 1.0])
                end_point_vis_opp_world = contact_point_world + force_direction_world * 0.05
            self.publish_arrow_oposite_world(start_point=contact_point_world,
                                             end_point=end_point_vis_opp_world,
                                             frame_id="world",
                                             marker_id=self.collision_marker_id_counter)
            self.publish_collision_point2(collided_link, contact_point_world, contact_point_world + force_direction_world*(-0.16),
                                                         frame_id_override="world",
                                                         marker_id=self.collision_marker_id_counter)



        else:
            rospy.logwarn(
                "Could not visualize contact estimation due to missing/invalid information (link, point, or force vector).")

    def create_sphere_bbox(self, MoveGroupArm, center, radius, name="collision_sphere"):
        """Creates a sphere bounding box as a collision object in MoveIt.

         Args:
             MoveGroupArm (MoveGroupPythonInterface): The MoveGroupPythonInterface object.
             center (list[float]): The center coordinates of the sphere [x, y, z].
             radius (float): The radius of the sphere.
             name (str, optional): The name of the collision object. Defaults to "collision_sphere".
         """
        # Prepare pose message
        ps = Pose()
        ps.position.x = center[0]
        ps.position.y = center[1]
        ps.position.z = center[2]
        ps.orientation.w = 1.0

        # Add the sphere to the planning scene
        MoveGroupArm.planning_scene.addSphere(name,radius, center[0], center[1], center[2])
        #rospy.loginfo(f"Added sphere collision object '{name}' to MoveIt.")
        return name

    def move_torques_check(self, velocity):
        rospy.loginfo("Starting movement...")
        # Initialize movement
        self.move_group.move_velocity(velocity)
        # Polling rate
        rate = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
            # Log current torques
            #rospy.loginfo(f"Current torques: {self.current_external_torques}")

            # Check if any joint torque exceeds the threshold
            if any(abs(force) > threshold for force, threshold in zip(self.current_external_torques, self.torque_thresholds)):
                # Stop movement
                self.move_group.move_velocity([0.0]*7)
                rospy.loginfo("Stopping movement, torque threshold reached...")
                collision_torques = self.current_external_torques.copy()

                # Isolate collision
                joint_angles = self.move_group.get_current_state()
                #print(self.robot.jacobe(joint_angles))
                print(f"JOINT ANGLES {joint_angles}")
                collision_torques, collided_link = self.isolate_collision_link(collision_torques)
                initial_s = self.initial_condition_selection(collided_link)

                # Use optimization-based collision point estimation
                collided_link_opt, contact_point_opt, force_estimated_opt, best_s_opt, force_amplitude = self.isolate_collision_point_optimization(collided_link,
                    initial_s, joint_angles, collision_torques)

                # Pass the world frame results directly to the updated visualization function
                self.visualize_contact_estimation(collided_link_opt, contact_point_opt, force_estimated_opt)

                break
            rate.sleep()

    def perform_single_measurement(self, timeout=10.0):
        """
        Monitors torques while the robot is STATIONARY and waits for an external
        contact event (torque threshold exceeded). If detected, estimates contact details.

        Args:
            timeout (float): Maximum time (seconds) to wait for an external contact event.

        Returns:
            tuple: Contains measurement results:
                (collided_link, contact_point, force_vector_base, optimized_s, link_length, force_amplitude)
        """
        start_time = rospy.Time.now()
        collision_detected = False
        collision_torques = None

        rate = rospy.Rate(50)  # 50 Hz check rate
        while not rospy.is_shutdown() and (rospy.Time.now() - start_time) < rospy.Duration(timeout):
            if any(abs(torque) > abs(thresh) for torque, thresh in
                   zip(self.current_external_torques, self.torque_thresholds)):
                collision_torques = self.current_external_torques.copy()
                rospy.loginfo(f"External contact detected! Torque threshold reached. Torques: {collision_torques}")
                collision_detected = True
                break

            rate.sleep()

        if not collision_detected:
            rospy.logwarn(f"No external contact detected within timeout ({timeout}s).")
            return None, None, None, None, None, None

        if collision_torques:
            joint_angles = self.move_group.get_current_state()

            collision_torques_diff, collided_link_initial = self.isolate_collision_link(collision_torques)
            if collided_link_initial is None:
                rospy.logerr("Failed to identify initial collided link from torque difference.")

            #  Get initial 's' guess
            initial_s = self.initial_condition_selection(collided_link_initial)

            collided_link_opt, contact_point_opt, force_estimated_base, best_s_opt, force_amplitude = self.isolate_collision_point_optimization(
                initial_s, joint_angles, collision_torques_diff)

            if collided_link_opt is not None:
                self.visualize_contact_estimation(collided_link_opt, contact_point_opt, force_estimated_base)
            else:
                rospy.logwarn("Optimization failed or returned invalid link, cannot visualize result.")
                return None, None, None, None, None, None

            absolute_s_distance = best_s_opt * self.total_length

            print(
                f"Measurement successful: Link={collided_link_opt}, s={best_s_opt:.4f}, LinkLength={absolute_s_distance:.4f}")
            return collided_link_opt, contact_point_opt, force_estimated_base, best_s_opt, absolute_s_distance, force_amplitude
        else:
            return None, None, None, None, None, None

    def measure_and_log(self, num_points=1, num_measurements_per_point=15, log_filename='measurement_log_point1.200.csv'):
        log_filepath = os.path.expanduser(log_filename)
        rospy.loginfo(
            f"Starting measurement process: {num_points} points, {num_measurements_per_point} measurements each.")
        rospy.loginfo(f"Logging data to: {log_filepath}")

        with open(log_filepath, 'w') as f:
            header = "Measured Point,Measurement Number,Optimized_s,Force_Amplitude,Collided_Link_Index,Link_Length_m\n"
            f.write(header)
            # Loop through the points of measurement
            for point in range(1, num_points + 1):

                rospy.loginfo(f"\n----- Measuring Point {point} out of {num_points} -----")
                input(
                    f"Measuring Point {point}, press Enter to begin measurements...")

                measurements = []  # List to store successful measurements for the current point
                measurement_num = 1
                while len(measurements) < num_measurements_per_point:
                    print(
                        f"---> Attempting Measurement {measurement_num} of {num_measurements_per_point} for Point {point}")

                    # Perform a single measurement
                    coll_link, _, _, opt_s, link_len, force_amplitude = self.perform_single_measurement()

                    if opt_s is not None and coll_link is not None:
                        print(
                            f"Measurement {measurement_num} successful: s={opt_s:.3f}, Force Amplitude={force_amplitude:.3f}, Link={coll_link}, Length={link_len:.3f}")
                        satisfied = input("Keep this measurement? (y/n): ").lower()
                        if satisfied == 'y':
                            measurements.append((opt_s,force_amplitude, coll_link, link_len))
                            log_line = f"{point},{len(measurements)},{opt_s:.3f},{force_amplitude:.3f},{coll_link},{link_len:.3f}\n"
                            f.write(log_line)
                            if len(measurements) < num_measurements_per_point:
                                input(
                                    f"Measurement {len(measurements)} recorded. Reset for next measurement and press Enter...")
                            measurement_num += 1  # Increment only after a successful and kept measurement
                        else:
                            rospy.loginfo("Measurement discarded, repeating.")
                            # measurement_num remains the same, so the next iteration repeats this attempt
                    else:
                        rospy.logwarn(f"Measurement {measurement_num} for Point {point} failed or timed out.")
                        retry = input("Retry measurement? (y/n): ").lower()
                        if retry == 'n':
                            rospy.loginfo("Skipping this measurement attempt.")
                            measurement_num += 1  # Move to the next measurement attempt even if failed

                rospy.loginfo(
                    f"Finished Point {point}. Successful measurements: {len(measurements)}/{num_measurements_per_point}")

            rospy.loginfo("----- Measurement Process Complete -----")
            rospy.loginfo(f"Log file saved to: {log_filepath}")

    def clear_all_collision_markers(self):
        rospy.loginfo("Clearing all previous collision markers...")
        # Delete all point markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "world"
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "collision_point"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        self.marker_pub.publish(delete_marker)

        # Delete all vector markers
        delete_marker.ns = "collision_vector_world"
        self.marker_pub.publish(delete_marker)
        delete_marker.ns = "collision_vector_opposite_world"
        self.marker_pub.publish(delete_marker)
        # Reset counter
        self.collision_marker_id_counter = 0

if __name__ == "__main__":
    # Threshold definition
    threshold_values = [5.0] * 7  # Set torque thresholds for each joint
    robot_motion = CollisionDetection(thresholds=threshold_values)
    movement_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Reduced velocity for testing
    try:
        # Main action in the program
        robot_motion.move_torques_check(velocity=movement_velocity)
        #robot_motion.measure_and_log(num_points = 1, num_measurements_per_point = 15)
    except rospy.ROSInterruptException:
        # Handle exceptions and stop the robot if interrupted
        robot_motion.move_group.move_velocity([0.0] * len(movement_velocity))  # Ensure to stop the robot
        rospy.loginfo("Robot stopped due to exception.")