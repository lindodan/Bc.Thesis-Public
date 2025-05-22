#!/usr/bin/env python3
import rospy
import numpy as np
import threading
import time
import os
import csv
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion # Added Point/Quaternion
from spatialmath import SE3

try:
    from optimization import CollisionDetection, loginfo_green # Import necessary items
except ImportError as e:
     print(f"Error importing CollisionDetection: {e}")
     print("Please ensure 'collision_detection_module.py' is in the Python path.")
     exit()

# Constants from ExploreTablePlane
RECOVERY_DISTANCE_Z_M = 0.05
COLLISION_CHECK_RATE_HZ = 50
MOVE_VELOCITY_SCALE = 0.01
MOVE_ACCEL_SCALE = 0.01

class ExploreTablePlane:
    def __init__(self, collision_detection):
        self.collision_detection = collision_detection

        # Access attributes directly from the interface instance
        self.move_group = self.collision_detection.move_group # Alias for convenience
        self.ee_link_name = self.move_group.eef_link
        self.active_joints = self.move_group.group.get_active_joints()
        self.planning_frame = self.move_group.planning_frame
        self.isolate_collision_link = self.collision_detection.isolate_collision_link
        self.initial_condition_selection = self.collision_detection.initial_condition_selection
        self.isolate_collision_point_optimization = self.collision_detection.isolate_collision_point_optimization
        self.visualize_contact_estimation = self.collision_detection.visualize_contact_estimation
        self.robot = self.collision_detection.robot
        self.collision_detection.clear_all_collision_markers()

        rospy.loginfo(f"ExploreTablePlane using MoveGroupPythonInterface for EE '{self.ee_link_name}' in frame '{self.planning_frame}'")

        # Define joint position states
        self.position_states = {
            "zero": [0, 0, 0, 0, 0, 0, 0],
            "home": [0, 0, 0, -1.57, 0, 0, 0],
            "table_q_right": [-0.58, 2.059, 0., 0.488, -1.57, 0.436, 0.], # right
            "table_q_left": [0.58, 2.059, 0., 0.488, -1.57, 0.436, 0.], # left
            "table_level" : [0., 0., 0., 0., 0., 0., 0.]
        }

        self.linkid= {1:"iiwa_link_1",
                   2:"iiwa_link_2",
                   3:"iiwa_link_3",
                   4:"iiwa_link_4",
                   5:"iiwa_link_5",
                   6:"iiwa_link_6",
                   7:"iiwa_link_7"}

        self.collision_event = threading.Event()
        self.stop_event = threading.Event()

        self.move_thread = None
        self.collision_check_thread = None

        self.joint_angles = None
        self.collision_torques = None

    def _set_speed(self, vel=MOVE_VELOCITY_SCALE, acc=MOVE_ACCEL_SCALE):
        """Sets the robot's speed and acceleration scaling using the interface methods."""
        self.move_group.set_max_velocity_scaling_factor(vel)
        self.move_group.set_max_acceleration_scaling_factor(acc)
        rospy.logdebug(f"Set speed scale to {vel}, accel scale to {acc} via interface.")


    def move_to_joint_state(self, q, description="target"):
        """
        Moves the robot to a target joint configuration (q) using the interface.
        Checks for collisions during the movement.
        Returns True if successful, False if interrupted by collision or MoveIt failure.
        """
        if self.stop_event.is_set():
            rospy.loginfo("Stop event set, not starting new move.")
            return False

        if len(q) != len(self.active_joints):
             rospy.logerr(f"Target joint state length ({len(q)}) does not match robot active joints ({len(self.active_joints)})")
             return False

        rospy.loginfo(f"Attempting to move to {description} joint state via interface: {[f'{x:.3f}' for x in q]}")
        self.collision_event.clear()

        self._set_speed()

        try:
            success = self.move_group.go_to_joint_position(q)
            rospy.loginfo(f"The real joint values {self.move_group.get_current_state()}")

        except Exception as e:
            rospy.logerr(f"Error during interface go_to_joint_position for {description}: {e}")
            self.move_group.stop_robot()
            return False

        # Check if a collision occurred *during* the execution
        if self.collision_event.is_set():
            rospy.logwarn(f"Movement to {description} interrupted by collision!")
            return False

        if not success:
            rospy.logwarn(f"Interface go_to_joint_position failed for {description}.")
            return False

        rospy.loginfo(f"Successfully reached {description} joint state via interface.")
        return True

    def check_collision_loop(self):
        """
        Continuously checks for collisions based on external torques from CollisionDetection.
        """
        rate = rospy.Rate(COLLISION_CHECK_RATE_HZ)
        rospy.loginfo("Starting collision check loop...")

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            try:
                current_torques = self.collision_detection.current_external_torques
                thresholds = self.collision_detection.torque_thresholds
                if len(current_torques) != len(thresholds):
                     rospy.logwarn_throttle(5, f"Torque ({len(current_torques)}) and threshold ({len(thresholds)}) list lengths differ.")
                     rate.sleep()
                     continue

                collision_detected_in_loop = False
                if any(abs(force) > threshold for force, threshold in zip(current_torques, thresholds)):
                    collision_torques = current_torques.copy()
                    collision_detected_in_loop = True

                if collision_detected_in_loop and not self.collision_event.is_set():
                    rospy.loginfo("Signaling collision event and stopping robot.")
                    self.move_group.stop_robot()
                    self.collision_event.set()
                    self.collision_torques = current_torques.copy()  # Save for later use
                    self.joint_angles = self.move_group.get_current_state()

            except AttributeError:
                 rospy.logerr_throttle(10,"CollisionDetection object missing required attributes.")
            except Exception as e:
                rospy.logerr(f"Error in collision check loop: {e}")

            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("Collision check loop interrupted.")
                break
        rospy.loginfo("Collision check loop finished.")


    def get_current_ee_pose(self):
        """Gets the current pose of the end-effector using the interface."""
        pose_current = self.move_group.get_ee_pose()
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "iiwa_link_0"
        ps.pose = pose_current
        if pose_current is None:
             rospy.logerr("Interface get_ee_pose() returned None.")
        return ps


    def move_link_up(self, distance_z=RECOVERY_DISTANCE_Z_M):
        """
        Moves the end-effector straight up using Cartesian path planning via the interface.
        """
        rospy.loginfo(f"Attempting to move link 5 up by {distance_z} m via interface.")
        q_current = self.move_group.get_current_state()
        pose_link_5 = self.robot.fkine(q_current,end = "iiwa_link_5")
        print("Original pose:\n", pose_link_5)

        pose_link_5 =  SE3.Tz(RECOVERY_DISTANCE_Z_M) * pose_link_5
        print("Translated pose:\n", pose_link_5)
        ik_sol = self.robot.ikine_LM(pose_link_5, q0=q_current[:5], end="iiwa_link_5", ilimit=40, slimit=500,
                                     joint_limits=True, tol=0.003)
        ik_q = ik_sol.q
        if len(ik_q) < len(q_current):
            ik_q_full = np.concatenate([ik_q, self.position_states["table_q_right"][len(ik_q):]])
        else:
            ik_q_full = ik_q
        try:
            self._set_speed(vel=0.05, acc=0.05)
            success = self.move_to_joint_state(ik_q_full,description="move link up")
            q_current = self.move_group.get_current_state()
            self.position_states["table_level"][1] = q_current[1]
            self.position_states["table_level"][2] = self.position_states["table_q_right"][2]
            self.position_states["table_level"][3] = q_current[3]
            self.position_states["table_level"][4] = q_current[4]
            self.position_states["table_level"][5] = self.position_states["table_q_right"][5]
            self.position_states["table_level"][6] = self.position_states["table_q_right"][6]


            if not success:
                rospy.logerr("Failed to execute Cartesian path for moving up via interface.")
                return False

            rospy.loginfo("Successfully moved link 5 up via interface.")
            return True

        except Exception as e:
            rospy.logerr(f"Error during Cartesian planning or execution for moving up via interface: {e}")
            self.move_group.stop_robot()
            return False

    def react_on_collision(self,move_back):
        """Handles the reaction sequence after a collision is detected."""

        rospy.logwarn("Collision reaction triggered...")
        # move robot a bit back
        q = self.move_group.get_current_state()
        q[0] += np.deg2rad(move_back)
        self.move_group.go_to_joint_position(q)
        rospy.sleep(0.3)
        if self.collision_torques is not None and self.joint_angles is not None:
            collision_torques, collided_link_initial= self.isolate_collision_link(self.collision_torques)
            initial_s = self.initial_condition_selection(collided_link_initial)
            collided_link_opt, contact_point_opt, force_estimated_opt, best_s_opt, force_amplitude = self.isolate_collision_point_optimization(collided_link_initial,
                initial_s, self.joint_angles, collision_torques)
            fault_isolation = abs(collided_link_opt - collided_link_initial) > 1
            self.visualize_contact_estimation(collided_link_opt, contact_point_opt, force_estimated_opt,fault_isolation)
        rospy.sleep(0.3)
        rospy.logwarn("Executing collision reaction: Moving up.")

        # clear the collision event
        self.collision_event.clear()
        success = self.move_link_up(distance_z=RECOVERY_DISTANCE_Z_M) # Uses updated move_link_up

        if success:
            rospy.loginfo("Recovery move successful.")
            self.collision_event.clear()
            rospy.sleep(0.3)
            return True
        else:
            if self.collision_event.is_set():
                rospy.logwarn("Collision detected during the recovery move (move_link_up).")
                self.collision_event.clear()
                rospy.sleep(0.3)  # Optional pause
                return True
            else:
                rospy.logerr(
                    "Recovery move failed (e.g., planning/execution error, no collision signal). Stopping exploration.")
                self.stop_event.set()
                return False

    def move_sequence(self):
        """The main sequence of movements for table exploration."""
        rospy.loginfo("Starting exploration movement sequence...")
        max_retries = 9
        retries = 0

        # --- Move 1 ---
        self.position_states["table_level"] = self.position_states["table_q_right"].copy()
        current_description = "table_plane_right"
        move1_success = False
        while retries <= max_retries and not self.stop_event.is_set():
            success = self.move_to_joint_state(self.position_states["table_level"], description=current_description)
            if success:
                rospy.loginfo(f"Reached {current_description} position.")
                move1_success = True
                break
            elif self.collision_event.is_set():
                if self.react_on_collision(8):
                     retries += 1
                     rospy.loginfo(f"Retrying move to {current_description} (Attempt {retries}/{max_retries})")
                else:
                     break
            else:
                 rospy.logerr(f"MoveIt failed to reach {current_description} (no collision signal). Aborting sequence.")
                 self.stop_event.set()
                 break
        # ... (check move1_success and stop_event) ...
        if not move1_success:
             rospy.logerr(f"Failed to complete move to {current_description} after {retries} retries/failures. Aborting sequence.")
             self.stop_event.set()

        if self.stop_event.is_set():
             rospy.loginfo("Stopping movement sequence.")
             return

        self.move_to_joint_state(self.position_states["table_q_right"], description="home_right")

        rospy.sleep(1.0)
        retries = 0
        self.position_states["table_level"] = self.position_states["table_q_left"].copy()
        current_description = "table_plane_left"
        move2_success = False
        while retries <= max_retries and not self.stop_event.is_set():
            success = self.move_to_joint_state(self.position_states["table_level"], description=current_description)

            if success:
                rospy.loginfo(f"Reached {current_description} position.")
                move2_success = True
                break
            elif self.collision_event.is_set():
                if self.react_on_collision(-8):
                     retries += 1
                     rospy.loginfo(f"Retrying move to {current_description} (Attempt {retries}/{max_retries})")
                else:
                     break # Recovery failed
            else:
                 rospy.logerr(f"MoveIt failed to reach {current_description} (no collision signal). Aborting sequence.")
                 self.stop_event.set()
                 break
        # ... (check move2_success) ...
        if not move2_success:
             rospy.logerr(f"Failed to complete move to {current_description} after {retries} retries/failures. Aborting sequence.")

        rospy.loginfo("Exploration movement sequence finished (or aborted).")
        self.move_to_joint_state(self.position_states["table_q_left"], description="home_left")


    def start(self):
        """Starts the exploration and collision monitoring threads."""
        # Logic remains the same
        if self.move_thread and self.move_thread.is_alive() or \
           self.collision_check_thread and self.collision_check_thread.is_alive():
            rospy.logwarn("Threads seem to be already running. Call shutdown() first if needed.")
            return

        rospy.loginfo("Initializing and starting threads...")
        self.stop_event.clear()
        self.collision_event.clear()
        self.contact_point_log_counter = 0
        self.collision_check_thread = threading.Thread(target=self.check_collision_loop, daemon=True)
        self.collision_check_thread.start()
        rospy.loginfo("Collision check thread started.")

        rospy.sleep(0.5)

        self.move_thread = threading.Thread(target=self.move_sequence)
        self.move_thread.start()
        rospy.loginfo("Movement sequence thread started.")

    def shutdown(self):
        """Cleans up resources and stops threads."""
        # Logic remains the same, but ensure stop_robot is called
        if self.stop_event.is_set():
             rospy.loginfo("Shutdown already in progress or completed.")
             return

        rospy.loginfo("Initiating shutdown sequence...")
        self.stop_event.set()

        if self.move_group:
            rospy.loginfo("Stopping robot motion via Interface...")
            try:
                 self.move_group.stop_robot()
            except Exception as e:
                 rospy.logerr(f"Error stopping move_group via interface: {e}")

        # ... (rest of thread joining logic is the same) ...
        if self.collision_check_thread and self.collision_check_thread.is_alive():
            rospy.loginfo("Waiting for collision check thread...")
            self.collision_check_thread.join(timeout=2.0)
            if self.collision_check_thread.is_alive(): rospy.logwarn("Collision check thread did not stop.")

        if self.move_thread and self.move_thread.is_alive():
            rospy.loginfo("Waiting for movement thread...")
            self.move_thread.join(timeout=5.0)
            if self.move_thread.is_alive(): rospy.logwarn("Movement thread did not stop.")

        rospy.loginfo("Exploration node shutdown complete.")


if __name__ == "__main__":
    node_name = 'explore_table_plane_node'
    try:
        rospy.init_node(node_name, anonymous=True)
        rospy.loginfo(f"'{node_name}' Node Started")

        threshold_values = [5.0] * 7
        rospy.loginfo(f"Using torque thresholds: {threshold_values}")

        try:
             # CollisionDetection will now instantiate the modified MoveGroupPythonInterface
             collision_detector = CollisionDetection(thresholds=threshold_values, buffer_size=10)
             rospy.loginfo("CollisionDetection instance created successfully.")
             rospy.sleep(1.0) # Allow time for internal initializations (subscriber, etc.)
        except NameError:
             rospy.logfatal("The 'CollisionDetection' class is not defined or imported correctly.")
             exit()
        except Exception as e:
             rospy.logfatal(f"Failed to initialize CollisionDetection: {e}")
             import traceback
             traceback.print_exc()
             exit()

        # --- Initialize and Start Exploration ---
        # ExploreTablePlane gets the collision_detector containing the modified interface
        explore_table = ExploreTablePlane(collision_detection=collision_detector)

        def shutdown_hook():
            rospy.logwarn(f"'{node_name}' received shutdown signal (e.g., Ctrl+C).")
            explore_table.shutdown()

        rospy.on_shutdown(shutdown_hook)

        explore_table.start()

        rospy.loginfo(f"'{node_name}' spinning. Exploration running in threads. Press Ctrl+C to stop.")
        rospy.spin()

    # ... (exception handling and finally block remain the same) ...
    except rospy.ROSInterruptException:
        rospy.loginfo(f"'{node_name}' interrupted by ROS. Shutting down.")
    except AttributeError as e:
         rospy.logfatal(f"Initialization Error: {e}. Check class attributes and method calls.")
         import traceback
         traceback.print_exc()
    except Exception as e:
        rospy.logfatal(f"An unexpected error occurred in '{node_name}': {e}")
        import traceback
        traceback.print_exc()
    finally:
         rospy.loginfo(f"'{node_name}' exiting.")