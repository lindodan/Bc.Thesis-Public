#!/usr/bin/env python3

import rospy
from kuka_humanoids.motion_interface import MoveGroupPythonInterface
from iiwa_msgs.msg import JointTorque
import numpy as np
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
        rospy.init_node("CollisionDetection_node")  # Initialize the ROS node

        # Initialize motion interface for robot
        self.move_group = MoveGroupPythonInterface("iiwa_arm")

        # Force threshold for each one of the joints
        self.torque_thresholds = thresholds

        # Initialize external forces (torques)
        self.current_external_torques = [0.0] * 7

        # Initialize robot model from robotics_toolbox with our URDF
        self.robot = ERobot.URDF("/home/docker/kuka_ws/src/iiwa_stack/iiwa_description/urdf/iiwa7.urdf.xacro")
        # Initialize torque buffer
        self.torque_buffer = []
        # Initialize size of torque buffer
        self.buffer_size = buffer_size

        # Subscribe to the external joint torque topic
        rospy.Subscriber("/iiwa/state/ExternalJointTorque", JointTorque, callback=self.torque_callback)

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
        """
        This will be probably highly dependent on moving speed
        """
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
                # On robot, we calculate links from 2
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

def example_run():
    # Threshold definition
    threshold_values = [5.0] * 7  # Set torque thresholds for each joint
    collision = CollisionDetection(thresholds=threshold_values)
    rospy.loginfo("Collision Detection example")

    # Polling rate
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # Log current torques
        rospy.loginfo(f"Current torques: {collision.current_external_torques}")

        # Check if any joint torque exceeds the threshold
        if any(abs(force) > threshold for force, threshold in
               zip(collision.current_external_torques, collision.torque_thresholds)):
            # Stop movement or do something else
            #self.move_group.move_velocity([0.0] * 7)
            rospy.loginfo("Stopping movement, torque threshold reached...")
            collision_torques = collision.current_external_torques.copy()

            # Isolate collision
            joint_angles = collision.move_group.get_current_state()
            print(f"JOINT ANGLES {joint_angles}")
            collision_torques, collision_link = collision.isolate_collision_link(collision_torques)
            break

        rate.sleep()

if __name__ == '__main__':
    try:
        example_run()
    except rospy.ROSInterruptException:
        rospy.logerr("Robot stopped due to exception.")