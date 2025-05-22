#!/usr/bin/env python3

import rospy
from kuka_humanoids.motion_interface import MoveGroupPythonInterface
from iiwa_msgs.msg import JointTorque
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
import numpy as np
import roboticstoolbox as rbt
from roboticstoolbox import ERobot

def loginfo_green(message):
    '''
    This function acts as a wrapper around rospy.loginfo to print messages in green color.
    '''
    green_start = '\033[92m'
    color_reset = '\033[0m'
    rospy.loginfo(green_start + str(message) + color_reset)

class CollisionDetection:
    def __init__(self, thresholds,buffer_size = 20):
        rospy.init_node("examples_node")  # Initialize the ROS node

        # Initialize motion interface for robot
        self.move_group = MoveGroupPythonInterface("iiwa_arm")

        # Force threshold for each one of the joints
        self.torque_thresholds = thresholds

        # Initialize external forces (torques)
        self.current_external_torques = [0.0] * 7

        # Initialize robot model from robotics_toolbox using our URDF
        self.robot = ERobot.URDF("/home/docker/kuka_ws/src/iiwa_stack/iiwa_description/urdf/iiwa7.urdf.xacro")
        # Initialize torque buffer
        self.torque_buffer = []
        # Initialize size of torque buffer
        self.buffer_size = buffer_size

        # Subscribe to the external joint torque topic
        rospy.Subscriber("/iiwa/state/ExternalJointTorque", JointTorque, callback=self.torque_callback)

        # Publisher for RViz
        self.marker_pub = rospy.Publisher("/iiwa/visualization_marker", Marker, queue_size=10)

    def torque_callback(self, msg):
        """
        Callback to update forces (torques) for each joint from the JointTorque message.
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
        #rospy.loginfo(f"Received torques: {self.current_external_torques}")

    def publish_collision_point(self,collided_link, point, marker_id=0):
        frameid = {1:"iiwa_link_1",
                   2:"iiwa_link_2",
                   3:"iiwa_link_3",
                   4:"iiwa_link_4",
                   5:"iiwa_link_5",
                   6:"iiwa_link_6",
                   7:"iiwa_link_7"}
        # Create a Marker object for visualizing the contact point
        marker = Marker()
        marker.header.frame_id = frameid[collided_link]
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_point"
        marker.id = marker_id
        marker.type = Marker.SPHERE  # Use a sphere to represent a single point
        marker.action = Marker.ADD

        marker.scale.x = 0.1  # Sphere radius
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.r = 0.0  # Red color
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Fully opaque

        # Add the point (ri_d) to the marker
        marker.pose.position = Point(*point)  # The ri_d point
        # No rotation quaternion
        marker.pose.orientation =Quaternion(0,0,0,1)

        # Publish the marker
        self.marker_pub.publish(marker)

    def publish_collision_line(self, collided_link, points, marker_id=0):
        frameid = {1: "iiwa_link_1",
                   2: "iiwa_link_2",
                   3: "iiwa_link_3",
                   4: "iiwa_link_4",
                   5: "iiwa_link_5",
                   6: "iiwa_link_6",
                   7: "iiwa_link_7"}
        # Create a Marker object for visualizing the line of action
        marker = Marker()
        marker.header.frame_id = frameid[collided_link]
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision_vector"
        marker.id = marker_id
        marker.type = Marker.ARROW  # Use LINE_STRIP to represent the line
        marker.action = Marker.ADD

        # Set the arrow's scale
        marker.scale.x = 0.03
        marker.scale.y = 0.05
        marker.scale.z = 0.05 # Head length

        marker.color.r = 0.91  # Pink color for the arrow
        marker.color.g = 0.12
        marker.color.b = 0.39
        marker.color.a = 1.0  # Fully opaque

        # Define the start and end points for the arrow
        if len(points) >= 2:
            start_point = Point(*points[0])
            end_point = Point(*points[-1])
        else:
            rospy.logerr("Insufficient points provided for collision arrow!")
            return

        marker.points = [start_point, end_point]  # Arrow defined by two points

        # Quaternion for no rotation
        #marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

        # Publish the marker
        self.marker_pub.publish(marker)

    def get_n_valid_torques(self,n):
        """
        Retrieves the n-th valid set of torques before the threshold was triggered.
        This function returns torque data from the buffer.
        """
        if self.torque_buffer and len(self.torque_buffer) >= n:
            return self.torque_buffer[n]

        # Default value
        return [0.0] * 7

    def isolate_collision_link(self, collision_torques):
        """
        Collision isolation algorithm to identify which link is in collision.
        The link that is in collision should be the one that has last non-zero value in collision_torques.

        collision_offset -> values of external torques in time =  buffer_size*rospy_rate(Hz) minus param of get_n_valid_torques.
        """
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


        # Apply the threshold to make small residuals zero
        #torque_difference = np.where(np.abs(torque_difference) < torque_threshold, 0.0, torque_difference)
        # Mask the collision torques
        collision_torques = np.where(np.abs(torque_difference) < torque_threshold, 0.0, collision_torques)

        # find the link in collision
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

        print(collision_torques)
        print(torque_difference)
        return collision_torques, collided_link

    def contact_jacobian(self,collided_link,J_i):
        """
        Makes the contact Jacobian. For a collision on link i_c, the last n−i_c columns
        of the contact Jacobian J_c(q) are identically zero.
        """
        #Create zero matrix for the trailing columns
        zero_cols = np.zeros((6,7-collided_link))

        # concatenate the J_i and zero_cols
        J_c = np.concatenate((J_i,zero_cols),axis=1)

        return J_c

    def isolate_collision_point(self,joint_angles,collision_torques,collided_link):
        """
        Collision isolation algorithm to identify which point  on robot is in collision.
        self.robot.links does not start with the fist link. The first one has idx 1
        Collided_link is also 1 for the 1 link
       """

        # Compute Jacobian for the collided link in collided link frame
        #self.robot.plot(joint_angles)
        try:
            J_i = self.robot.jacobe(joint_angles, end=self.robot.links[collided_link+1])
            rospy.loginfo(f"Jacobian shape for link {collided_link}: {J_i.shape}")
        except Exception as e:
            rospy.logwarn(f"Failed to compute Jacobian for link {collided_link}: {e}")

        # Get J_i jacobian (6x7) add zeros so its full size
        #J_i = self.contact_jacobian(collided_link,J_i)

        rospy.loginfo(f"\n Collision link number {collided_link} jacobian: {J_i}")
        # Calculate external force
        # (6 x i_c) @ (i_c x 1) -> (6x1)
        F_i = np.linalg.pinv(J_i.T) @ collision_torques[:collided_link]
        print(f"F_ext: {F_i}")

        # Extract force and moment part from F_ext
        f_i = F_i[:3] # only the force part its the contact force vector ?
        m_i = F_i[3:] # moment
        print(f"fi {f_i}")
        print(f"mi {m_i}")
        m_i_normalized = m_i/np.linalg.norm(m_i)
        print(f"nomralized mi {m_i_normalized}")
        # Skew-symm matrix for the external forces  (3x3)
        S_f_i = np.array([
            [0, -f_i[2], f_i[1]],
            [f_i[2], 0, -f_i[0]],
            [-f_i[1], f_i[0], 0]
        ])
        print(f"RANK OF MATRIX S_f_i {np.linalg.matrix_rank(S_f_i)}")
        # solve for the minimum distance vector
        # (3x3) @ (3x1) -> (3x1)
        ri_d= np.linalg.pinv(S_f_i.T) @ m_i_normalized #in the link i frame ?
        print(ri_d)
        # makes a line
        f_i_normalized = f_i/np.linalg.norm(f_i)
        lambdas = np.linspace(-0.3,0.3,15)
        points_local = [ri_d + lam * f_i_normalized for lam in lambdas]

        # Calculate transforamtion from collided link to base
        T = self.robot.fkine(joint_angles,end=self.robot.links[collided_link+1])

        # Transform the point from local to global
        #points_global = [T[:3, :3] @ p + T[:3, 3] for p in points_local]

        # Visualize as line list
        self.publish_collision_point(collided_link,ri_d)
        self.publish_collision_line(collided_link,points_local)
        rospy.loginfo(f"Approximate contact point: {ri_d} on a link {collided_link}")
        return collided_link, ri_d

    def move_torques_check(self, velocity):
        rospy.loginfo("Starting movement...")
        # Initialize movement
        self.move_group.move_velocity(velocity)
        # Polling rate
        rate = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
            # Log current torques
            rospy.loginfo(f"Current torques: {self.current_external_torques}")

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
                collision_torques,collision_link = self.isolate_collision_link(collision_torques)
                self.isolate_collision_point(joint_angles,collision_torques,collision_link)


                break

            rate.sleep()

if __name__ == "__main__":
    # Threshold definition
    threshold_values = [5.0] * 7  # Set torque thresholds for each joint
    robot_motion = CollisionDetection(thresholds=threshold_values)
    movement_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        # Main action in the program
        q = robot_motion.move_group.get_current_state()
        print(f"q: {q}")
        #robot_motion.move_torques_check(velocity=movement_velocity)
    except rospy.ROSInterruptException:
        # Handle exceptions and stop the robot if interrupted
        robot_motion.move_group.move_velocity([0.0] * len(movement_velocity))  # Ensure to stop the robot
        rospy.loginfo("Robot stopped due to exception.")

