#!/usr/bin/env python3

# Imports from your original code + Marker
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
import rospy
from kuka_humanoids.motion_interface import MoveGroupPythonInterface
import numpy as np
import time # For sleep
from visualization_msgs.msg import Marker # Import Marker

if __name__ == '__main__':
    # Initialize node (use anonymous=True to prevent node name conflicts if run multiple times)
    rospy.init_node("examples_node", anonymous=True)

    # Your MoveGroup interface (not strictly needed for just publishing a marker, but kept from your code)
    # MoveGroupArm = MoveGroupPythonInterface("iiwa_arm")

    # Publisher for the marker
    marker_pub = rospy.Publisher("/iiwa/visualization_marker", Marker, queue_size=10)

    # Box parameters (only position is used for the marker pose in this version)
    collision_box_name = "collision_box"
    box_dimensions = (0.16, 0.16, 0.165)

    box_center_position = (0.565, 0.0, 0.165/2)

    # Wait briefly for publisher to connect (optional but good practice)
    rospy.sleep(0.5)

    # --- Create and configure the Marker ---
    marker = Marker()
    marker.header.frame_id = "iiwa_link_0" # Or use MoveGroupArm.get_planning_frame() if you need it relative to that
    marker.header.stamp = rospy.Time.now()
    marker.ns = "collision_box_left"
    marker.id = 1
    marker.type = Marker.CUBE
    marker.action = Marker.ADD


    marker.pose.position.x = box_center_position[0]
    marker.pose.position.y = box_center_position[1]
    marker.pose.position.z = box_center_position[2] -0.025

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = box_dimensions[0]
    marker.scale.y = box_dimensions[1]
    marker.scale.z = box_dimensions[2]


    marker.color.r = 0
    marker.color.g = 255/255
    marker.color.b = 0
    marker.color.a = 0.7 # Semi-transparent

    # --- Publish the Marker ---
    rospy.loginfo(f"Publishing CUBE marker at {box_center_position} with scale {marker.scale.x, marker.scale.y, marker.scale.z}")
    marker_pub.publish(marker)