#!/usr/bin/env python3
import rospy
from kuka_humanoids.motion_interface import MoveGroupPythonInterface
from iiwa_msgs.msg import JointTorque
import numpy as np
import time
import threading


class PickAndPlaceRobot:
    def __init__(self,torque_threshold):
        # Initialize ROS node
        rospy.init_node('pick_and_place_node')

        # Initialize move group interface
        self.move_group = MoveGroupPythonInterface("iiwa_arm")

        # Initialize current external torques to zero
        self.current_external_torques = [0.0] * 7
        self.stop_execution = False

        # Define joint position states
        self.position_states = {
            "zero": [0, 0, 0, 0, 0, 0, 0],
            "home": [0, 0, 0, -1.57, 0, 0, 0],
            "pick_transfer": [-0.26112, 0.85609, 0.00037, -1.21273, 0.00119, 1.02233, -0.32047],
            "pick": [-0.26138, 1.06701, 0.00037, -1.21418, 0.00136, 0.81040, -0.32086],
            "place_transfer": [0.32347, 0.80991, 0.00038, -1.10696, -0.02919, 1.18231, 0.27529],
            "place": [0.32346, 1.15182, 0.00038, -1.13947, -0.03726, 0.80804, 0.28998],
            "bottom_sweep" : [0., 2.05948852, 0., 0.48869219, 0., 1.0471975512, 0.]

        }

        # Set torque thresholds
        self.torque_threshold = torque_threshold

        # Subscribe to external joint torque topic
        rospy.Subscriber("/iiwa/state/ExternalJointTorque", JointTorque, self.torque_callback)

    def torque_callback(self, data):
        # Update current external torques
        self.current_external_torques = [
            data.torque.a1, data.torque.a2, data.torque.a3,
            data.torque.a4, data.torque.a5, data.torque.a6, data.torque.a7
        ]

    def move_to_joint_state(self, joint_values):
        # Move the robot to the desired joint configuration
        self.move_group.go_to_joint_position(joint_values)

    def pick_and_place(self):
        sequence = ['pick_transfer', 'pick', 'place_transfer', 'place', 'pick_transfer']
        for pose in sequence:
            if self.stop_execution:
                break
            joint_values = self.position_states[pose]
            rospy.loginfo(f"Moving to {pose} with joint values {joint_values}")
            self.move_to_joint_state(joint_values)
            time.sleep(0.5)
        rospy.loginfo("Pick and place sequence completed")

    def stop_operation(self):
        rate = rospy.Rate(500)  # 500 Hz
        while not rospy.is_shutdown():
            # Log current torques
            rospy.loginfo(f"Current torques: {self.current_external_torques}")

            # Check if any torque exceeds the threshold
            if any(abs(torque) > threshold for torque, threshold in
                   zip(self.current_external_torques, self.torque_threshold)):
                self.stop_execution = True
                self.move_group.stop_robot()
                rospy.loginfo("Torque threshold exceeded! Stopping robot.")
                break

            rate.sleep()

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


if __name__ == "__main__":
    try:
        torque_threshold = [5.0] * 7
        robot = PickAndPlaceRobot(torque_threshold)
        robot.start()
    except rospy.ROSInterruptException:
        pass
