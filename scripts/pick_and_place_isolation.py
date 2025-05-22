#!/usr/bin/env python3

import rospy
from kuka_humanoids.motion_interface import MoveGroupPythonInterface
from iiwa_msgs.msg import JointTorque
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
import numpy as np
import roboticstoolbox as rbt
from roboticstoolbox import ERobot
from roboticstoolbox import Link, ET
from scipy.optimize import minimize, differential_evolution, direct
import time
import threading

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
        buffer_size (int, optional): The size of the torque buffer. Defaults to 20.
    """
    def __init__(self, thresholds, buffer_size = 20):
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

        # Link lengths approximation (from URDF visual inspection and some educated guesses)
        self.link_lengths = [
            0.15, # link 0
            0.19, # link 1
            0.21, # link 2
            0.19, # link 3
            0.21, # link 4
            0.19, # link 5
            0.0607, # link 6
            0.045 # link 7
        ]
        self.total_length = sum(self.link_lengths)

        # Keep track of added collision objects
        self.added_collision_objects = []
        self.collision_object_counter = 0

        # Subscribe to the external joint torque topic
        rospy.Subscriber("/iiwa/state/ExternalJointTorque", JointTorque, callback=self.torque_callback)

        # Publisher for RViz
        self.marker_pub = rospy.Publisher("/iiwa/visualization_marker", Marker, queue_size=10)

        # Define joint position states
        self.position_states = {
            "zero": [0, 0, 0, 0, 0, 0, 0],
            "home": [0, 0, 0, -1.57, 0, 0, 0],
            "pick_transfer": [-0.26112, 0.85609, 0.00037, -1.21273, 0.00119, 1.02233, -0.32047],
            "pick": [-0.26138, 1.06701, 0.00037, -1.21418, 0.00136, 0.81040, -0.32086],
            "place_transfer": [0.32347, 0.80991, 0.00038, -1.10696, -0.02919, 1.18231, 0.27529],
            "place": [0.32346, 1.15182, 0.00038, -1.13947, -0.03726, 0.80804, 0.28998]
        }
    def move_to_joint_state(self, joint_values):
        # Move the robot to the desired joint configuration
        self.move_group.go_to_joint_position(joint_values)

    def torque_callback(self, msg):
        """Callback function to update the current external joint torques.

        Args:
            msg (iiwa_msgs.msg.JointTorque): The received ExternalJointTorque message.
        """

        new_torques = [
            msg.torque.a1, msg.torque.a2, msg.torque.a3,
            msg.torque.a4, msg.torque.a5, msg.torque.a6, msg.torque.a7
        ]

        # Add new torques to buffer
        self.torque_buffer.append(new_torques)

        # Limit buffer to buffer size
        if len(self.torque_buffer) > self.buffer_size:
            self.torque_buffer.pop(0)

        # Update current torques
        self.current_external_torques = new_torques
        # rospy.loginfo(f"Received torques: {self.current_external_torques}")

    def publish_collision_point(self, collided_link, point, force_vector, marker_id=0, frame_id_override=None):
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

        marker.color.r = 1.0  # Red color
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.7

        """        # Shift the point origin in the negative direction of the force vector by 0.08
        force_norm = np.linalg.norm(force_vector)
        if force_norm > 1e-6:  # Avoid division by zero
            direction = force_vector / force_norm
            shift = -0.08 * direction
            shifted_point = np.array(point) + shift
            marker.pose.position = Point(*shifted_point)  # The shifted point
        else:
            marker.pose.position = Point(*point)  # Use original point if force is zero
            rospy.logwarn("Estimated force vector is close to zero, not shifting collision point 2.")"""
        # No rotation quaternion
        marker.pose.position = Point(*point)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        # Publish the marker
        self.marker_pub.publish(marker)

        # Add the same sphere as a collision object to MoveIt
        self.create_sphere_bbox(
            MoveGroupArm=self.move_group,
            center=point,
            radius=marker.scale.x / 2.0,  # Use radius (scale is diameter)
            name="collision_sphere"
        )

        time.sleep(8)
        self.remove_sphere_bbox(self.move_group)

        # Publish the marker
        self.marker_pub.publish(marker)

    def remove_sphere_bbox(self,MoveGroupArm, name="collision_sphere"):
        """Removes a sphere collision object from the MoveIt planning scene.

        Args:
            MoveGroupArm (MoveGroupPythonInterface): The MoveGroupPythonInterface object.
            name (str, optional): The name of the collision object to remove.
        """
        MoveGroupArm.planning_scene.removeCollisionObject(name)
        rospy.loginfo(f"Removed collision object '{name}' from MoveIt.")

    def publish_collision_line(self, collided_link, points, marker_id=1, frame_id_override=None):
        """Publishes an arrow marker to RViz representing a collision force vector.

        Args:
            collided_link (int): The index of the link in collision.
            points (list[list[float]]): A list containing the start and end points of the line.
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
        marker_frame_id = frame_id_override if frame_id_override else frameid[collided_link]
        print(marker_frame_id)
        # Create a Marker object for visualizing the line of action
        marker = Marker()
        marker.header.frame_id = marker_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_vector"
        marker.id = marker_id
        marker.type = Marker.ARROW  # Use ARROW to represent the force vector
        marker.action = Marker.ADD

        # Set the arrow's scale
        marker.scale.x = 0.02  # Reduced thickness
        marker.scale.y = 0.04
        marker.scale.z = 0.04  # Head length

        marker.color.r = 0.0  # Pink color for the arrow
        marker.color.g = 0.1
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        # Define the start and end points for the arrow
        if len(points) >= 2:
            start_point = Point(*points[0])
            end_point = Point(*points[-1])
        else:
            rospy.logerr("Insufficient points provided for collision arrow!")
            return

        marker.points = [start_point, end_point]  # Arrow defined by two points
        marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        # Publish the marker
        self.marker_pub.publish(marker)

    def get_n_valid_torques(self, n):
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

        # TODO: CHeck the function after the torques reading from robot are fixed
        collided_link = None

        # Lower index is older value
        collision_offset = self.get_n_valid_torques(0)

        # These are the torques from the collision when we exceed the threshold
        print(f"COLLISION TORQUES: {collision_torques}")
        # Thees are the pre-collision torques
        print(f"COLLISION OFFSET: {collision_offset}")

        # Calculate the difference between the collision torques and the collision offset
        torque_difference = np.subtract(collision_torques, collision_offset)

        # Threshold to zero residuals
        torque_threshold = 0.005

        test_torques = collision_torques  # CHECK THIS !!!!!
        # Apply the threshold to make small residuals zero
        collision_torques = np.where(np.abs(torque_difference) < torque_threshold, 0.0, collision_torques)

        # find the link in collision (first 0 value)
        for i, torque in enumerate(collision_torques):
            if abs(torque) > 0.0:
                # On robot we calculate links from 2
                collided_link = i + 1
            else:
                break
        if collided_link is not None:
            loginfo_green(f"Collided link index: {collided_link}")
        else:
            rospy.logerr("Collided link was not found")
            # TODO : Check this part
            collided_link = 4
        # Mask the torques that follow so they have also a 0.0 value
        if collided_link < len(collision_torques - 1):
            masked_collision_torques = np.copy(collision_torques)
            masked_collision_torques[collided_link:] = 0.0
            collision_torques = masked_collision_torques

        # return collision_torques, collided_link
        return test_torques, collided_link

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
        #print(s_total_length)

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
            collided_link_index = 7 # Default to link 7 if s is slightly out of range (due to numerical issues)
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
        iiwa_ets_list.append(ET.tz(0.0607) * ET.Rz(jindex=6) * ET.tz(0.045))  # link 7
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

        if plotting:
            new_robot_ets.plot(q[:collided_link_index],block = False, backend = 'pyplot')

        # Calculate FKine and Jacobian using the new robot model
        T_link_base = new_robot_ets.fkine(coords)  # FKine to the end of the ETS chain (collision point)
        po_base_frame = T_link_base.t

        jacobian_po_base = new_robot_ets.jacob0(coords)  # Jacobian to the end of the ETS chain

        return jacobian_po_base, po_base_frame, collided_link_index

    def objective_function_kuka(self, x, q, tau_f_measured):
        """Objective function for the optimization to estimate contact force and location.

        Args:
            x (list[float]): Optimization variables [s, phi], where s is the normalized position and phi is the force angle.
            q (numpy.ndarray): The current joint angles of the robot (7x1).
            tau_f_measured (numpy.ndarray): The measured external joint torques (7x1).

        Returns:
            float: The objective function value.
        """

        s_val = x[0]
        phi = x[1]
        f_bar_s = np.array([np.cos(phi), np.sin(phi), 0])  # Trigonometric identity
        jacobian_po_s, _, collided_link = self.get_link_jacobian_and_point(q, s_val)

        # Reduce Jacobian columns based on collided link (Eq. 8 in paper)
        jacobian_po_s_reduced = jacobian_po_s[:, :collided_link]
        zero_cols = np.zeros((6, 7 - collided_link))
        jacobian_po_s_contact = np.concatenate((jacobian_po_s_reduced, zero_cols), axis=1)

        # Use only positional Jacobian part (3x7)
        tau_bar_s = jacobian_po_s_contact[:3, :].T @ f_bar_s  # (Eq. 17 in paper)  (7x3) @ (3x1) -> (7x1)
        tau_bar_s_norm = np.linalg.norm(tau_bar_s)
        if tau_bar_s_norm < 1e-9:  # Handle near-zero norm
            return 1e6

        # Normalize the measured external torques
        tau_f_measured_norm = tau_f_measured / np.linalg.norm(tau_f_measured)

        tau_bar_s_normalized = tau_bar_s / tau_f_measured_norm

        objective_val = np.linalg.norm(tau_f_measured_norm - tau_bar_s_normalized)  # (Eq. 27 in paper)
        return objective_val

    def isolate_collision_point_optimization(self, joint_angles, collision_torques):
        """Estimates the collision point and force using a two-stage optimization process.

        Args:
            joint_angles (list[float]): The current joint angles of the robot.
            collision_torques (list[float]): The measured external joint torques.

        Returns:
            tuple[int, numpy.ndarray, numpy.ndarray]: A tuple containing the index of the collided link (1-7), the estimated contact point in the base frame (3x1), and the estimated contact force in the link frame (3x1). Returns (None, None, None) if optimization fails.
        """

        q = joint_angles
        tau_f_measured = np.array(collision_torques)

        # Define bounds for s (normalized position along the robot) and phi
        bounds = [(0.0, 1.0), (0.0, 2 * np.pi)]  # s is between 0 and 1 and phi is 0 and 2pi

        # Initial guess for s and phi
        initial_guess = [0.50, 0.0]

        # --- Global Optimization (DIRECT) --- solid parameters 26.03.2025
        rospy.loginfo("Starting global optimization (DIRECT)...")
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
        rospy.loginfo(f"Global optimization took: {duration_global:.6f} seconds")

        if hasattr(global_optimization_result, 'success') and global_optimization_result.success:
            best_global_x = global_optimization_result.x
            rospy.loginfo(
                f"Global optimization 2 successful. Best [s, phi]: {best_global_x}, Objective value: {global_optimization_result.fun}")
            initial_guess = best_global_x  # Use the global optimum as the starting point for local optimization
        else:
            rospy.logwarn(
                f"Global optimization failed: {getattr(global_optimization_result, 'message', 'No message')}, using initial guess for local optimization.")

        # --- Local Optimization (SLSQP) ---
        rospy.loginfo("Starting local optimization (SLSQP)...")
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
        rospy.loginfo(f"Local SLSQP optimization took: {duration_local:.6f} seconds")

        if hasattr(optimization_result, 'success') and optimization_result.success:
            best_s_phi = optimization_result.x
            best_s = best_s_phi[0]
            best_phi = best_s_phi[1]
            min_objective_value = optimization_result.fun
            rospy.loginfo(
                f"Local optimization 2 successful. Best [s, phi]: {best_s_phi}, Objective value: {min_objective_value}")

            jacobian_po_best_s, po_best_s, collided_link = self.get_link_jacobian_and_point(q, best_s)
            if jacobian_po_best_s is None:
                rospy.logwarn("Could not get Jacobian for the best s.")
                return None, None, None

            # Reduced Jacobian for force estimation
            jacobian_po_s_reduced = jacobian_po_best_s[:, :collided_link]
            zero_cols = np.zeros((6, 7 - collided_link))
            jacobian_po_s_contact = np.concatenate((jacobian_po_s_reduced, zero_cols), axis=1)

            f_bar_s_estimated = np.array([np.cos(best_phi), np.sin(best_phi), 0])
            tau_bar_s = jacobian_po_s_contact[:3, :].T @ f_bar_s_estimated
            tau_bar_s_norm = np.linalg.norm(tau_bar_s)

            if tau_bar_s_norm > 1e-9:
                f_s_magnitude_estimated = np.linalg.norm(tau_f_measured) / tau_bar_s_norm
                f_s_estimated = f_s_magnitude_estimated * f_bar_s_estimated
            else:
                rospy.logwarn("Norm of estimated torque direction is too small.")
                f_s_estimated = np.array()

            rospy.loginfo(f"Estimated contact force 2 in link frame: {f_s_estimated}")
            rospy.loginfo(f"Estimated contact point 2 in base frame: {po_best_s},{po_best_s.type}")
            rospy.loginfo(f"Measured torques: {tau_f_measured}")

            return collided_link, po_best_s, f_s_estimated

        else:
            rospy.logwarn(f"Local optimization failed: {getattr(optimization_result, 'message', 'No message')}")
            return None, None, None
    def visualize_contact_estimation(self, collided_link, contact_point, force_vector,collision_number):
        """Publishes RViz markers to visualize the estimated contact point and force vector.

        Args:
            collided_link (int): The index of the link in collision (1-7).
            contact_point (numpy.ndarray): The estimated contact point coordinates in the base frame (3x1).
            force_vector (numpy.ndarray): The estimated contact force vector in the link frame (3x1).
            collision_number (int): The number of the collision in time.
        """

        if collided_link is not None and contact_point is not None and force_vector.size > 0:
            self.publish_collision_point(collided_link, contact_point, force_vector, marker_id=collision_number,
                                          frame_id_override="world")
            force_vector_end_point = contact_point + force_vector / 100.0  # Scale force for visualization
            self.publish_collision_line(collided_link, [contact_point, force_vector_end_point],
                                         frame_id_override="world")
        else:
            rospy.logwarn("Could not visualize contact estimation due to missing information.")

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
        rospy.loginfo(f"Added sphere collision object '{name}' to MoveIt.")

    def start(self):
        # Start the pick and place operation in a separate thread
        pick_and_place_thread = threading.Thread(target=self.pick_and_place)
        pick_and_place_thread.start()

        # Start the stop operation thread to monitor torques
        stop_thread = threading.Thread(target=self.stop_operation)
        stop_thread.start()

        # Wait for both threads to finish
        pick_and_place_thread.join()
        stop_thread.join()

    def stop_operation(self):
        rate = rospy.Rate(60)  # 100 Hz
        collision_number = 0
        while not rospy.is_shutdown():
            # Log current torques
            #rospy.loginfo(f"Current torques: {self.current_external_torques}")

            # Check if any torque exceeds the threshold
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

                # Use optimization-based collision point estimation
                #self.isolate_collision_point_optimization(joint_angles, collision_torques)
                # Use optimization-based collision point estimation
                collided_link_opt, contact_point_opt, force_estimated_opt = self.isolate_collision_point_optimization(joint_angles, collision_torques)
                self.visualize_contact_estimation(collided_link_opt, contact_point_opt, force_estimated_opt)

                if self.stop_execution:
                    break


            rate.sleep()

    def pick_and_place(self):
        sequence = ['pick_transfer', 'pick', 'place_transfer', 'place', 'pick_transfer']
        for _ in range(3):
            for pose in sequence:
                #if self.stop_execution:
                    #break
                joint_values = self.position_states[pose]
                rospy.loginfo(f"Moving to {pose} with joint values {joint_values}")
                self.move_to_joint_state(joint_values)
                time.sleep(0.5)
        rospy.loginfo("Pick and place sequence completed")
        self.stop_execution = True

if __name__ == "__main__":
    # Threshold definition
    threshold_values = [5.0] * 7  # Set torque thresholds for each joint
    robot_motion = CollisionDetection(thresholds=threshold_values)
    movement_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Reduced velocity for testing
    try:
        # Main action in the program
        robot_motion.start()
    except rospy.ROSInterruptException:
        # Handle exceptions and stop the robot if interrupted
        robot_motion.move_group.move_velocity([0.0] * len(movement_velocity))  # Ensure to stop the robot
        rospy.loginfo("Robot stopped due to exception.")