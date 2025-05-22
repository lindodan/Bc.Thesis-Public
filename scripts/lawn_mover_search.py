#!/usr/bin/env python3
import rospy
import numpy as np
import threading
import time
import os
import csv # <--- ADD THIS IMPORT
from spatialmath import SE3, SO3
import math

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from optimization import CollisionDetection, loginfo_green

# Constants
RECOVERY_DISTANCE_Z_M = 0.05
RECOVERY_DISTANCE_Y_M = 0.05 # Distance to move back
WAYPOINT_DISTANCE_M = 0.05 # Sample trajectory every 5 cm
MAX_RAISE_RETRIES = 3       # Max attempts to move raised before failing segment
COLLISION_CHECK_RATE_HZ = 50
DEFAULT_MOVE_VELOCITY_SCALE = 0.01
DEFAULT_MOVE_ACCEL_SCALE = 0.01

class ExploreTablePlane:
    def __init__(self, collision_detection,
                 search_x_min=0.5, search_x_max=0.7,
                 search_y_min=-0.3, search_y_max=0.3,
                 search_z_height=0.2, search_step_size=0.1,
                 search_orientation=Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)):

        self.collision_detection = collision_detection

        self.move_group = self.collision_detection.move_group
        self.ee_link_name = self.move_group.eef_link
        self.planning_frame = self.move_group.planning_frame
        self.robot = self.collision_detection.robot
        self.isolate_collision_link = self.collision_detection.isolate_collision_link
        self.initial_condition_selection = self.collision_detection.initial_condition_selection
        self.isolate_collision_point_optimization = self.collision_detection.isolate_collision_point_optimization
        self.collision_marker_id_counter = self.collision_detection.collision_marker_id_counter
        self.publish_collision_point = self.collision_detection.publish_collision_point
        self.publish_arrow_world = self.collision_detection.publish_arrow_world
        self.publish_arrow_oposite_world = self.collision_detection.publish_arrow_oposite_world
        self.publish_collision_point2 = self.collision_detection.publish_collision_point_eef
        self.collision_detection.clear_all_collision_markers()
        rospy.loginfo(f"ExploreTablePlane using MoveGroup interface for EE '{self.ee_link_name}' in frame '{self.planning_frame}'")

        self.search_x_min = search_x_min
        self.search_x_max = search_x_max
        self.search_y_min = search_y_min
        self.search_y_max = search_y_max
        self.search_z_height = search_z_height # Target Z for the scan
        self.search_step_size = search_step_size # Step along X
        self.search_orientation = search_orientation

        self.position_states = {
            "home": [0, 0, 0, -1.57, 0, 1.57, 0.78], # Example KUKA home
             "safe_up" : [0, 0, 0, -1.57, 0, 1.57, 0] # Example raised pose
        }

        self.collision_event = threading.Event()
        self.stop_event = threading.Event()

        self.move_thread = None
        self.collision_check_thread = None

        self.joint_angles_on_collision = None
        self.collision_torques = None
        self.current_segment_start_pose = None
        self.current_segment_end_pose = None
        self.output_csv_filename = "measurement_right_EEF/contact_points_log3_right.csv"
        self.contact_point_log_counter = 0

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


    def _set_speed(self, vel=DEFAULT_MOVE_VELOCITY_SCALE, acc=DEFAULT_MOVE_ACCEL_SCALE):
        """Sets the robot's speed and acceleration scaling using the interface methods."""
        self.move_group.set_max_velocity_scaling_factor(vel)
        self.move_group.set_max_acceleration_scaling_factor(acc)

    def _get_current_joint_values(self):
        """Gets the current joint values using the interface."""
        return self.move_group.get_current_state()

    def move_to_joint_state(self, joint_goal, description="target joint state"):
        """Moves the robot to a target joint configuration."""
        if self.stop_event.is_set():
            rospy.loginfo(f"Stop event set, skipping move: {description}")
            return False
        rospy.loginfo(f"Attempting joint move: {description}...")
        try:
            self.move_group.go_to_joint_position(joint_goal)
            self.move_group.stop_robot()


            if self.collision_event.is_set():
                rospy.logwarn(f"Collision detected during/after joint move: {description}")
                return False # Indicate collision occurred

            rospy.loginfo(f"Joint move '{description}' completed.")
            return True # Move successful

        except Exception as e:
            rospy.logerr(f"Exception during move_to_joint_state for {description}: {e}")
            try: self.move_group.stop_robot()
            except: pass
            return False

    def visualize_contact_estimation(self, collided_link, contact_point_world, force_vector_world,fault_isolation = False,side = 0):
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
            shift = 0
            if side%2 == 1:
                shift =-0.16
            else:
                shift =+0.16
            contact_point_2 = contact_point_world +[0.,shift,0.]

            self.publish_collision_point2(collided_link, contact_point_world, contact_point_2,
                                                         frame_id_override="world",
                                                         marker_id=self.collision_marker_id_counter)
            self.contact_point_log_counter += 1
            self.log_contact_point_to_csv(self.contact_point_log_counter,contact_point_2)

        else:
            rospy.logwarn(
                "Could not visualize contact estimation due to missing/invalid information (link, point, or force vector).")


    def move_to_pose(self, target_pose, description="target pose", velocity_scale=None, acceleration_scale=None):
        """Moves the robot's EEF to a target Cartesian pose."""
        if self.stop_event.is_set():
            rospy.loginfo(f"Stop event set, skipping move: {description}")
            return False

        if not isinstance(target_pose, Pose):
            rospy.logerr(f"Invalid target_pose type for {description}. Expected Pose.")
            return False

        self._set_speed() # Use defaults

        rospy.loginfo(f"Attempting pose move: {description} to P(x={target_pose.position.x:.3f}, y={target_pose.position.y:.3f}, z={target_pose.position.z:.3f})")
        success = True
        try:
            self.move_group.go_to_pose(target_pose, wait=True)

        except Exception as e:
            rospy.logerr(f"Exception during move_to_pose for {description}: {e}")
            try: self.move_group.stop_robot()
            except: pass
            success = False


        if self.collision_event.is_set():
            rospy.logwarn(f"Collision detected during/after pose move: {description}")
            return False # Indicate collision occurred

        if not success:
            rospy.logerr(f"Pose move '{description}' failed (not collision related).")
            return False

        rospy.loginfo(f"Pose move '{description}' completed.")
        return True # Move successful


    def check_collision_loop(self):
        """Continuously checks for collisions based on external torques."""
        rate = rospy.Rate(COLLISION_CHECK_RATE_HZ)
        rospy.loginfo("Starting collision check loop...")
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            collision_detected_in_loop = False
            current_torques = self.collision_detection.current_external_torques
            thresholds = self.collision_detection.torque_thresholds
            # --- End Direct Access ---

            if current_torques is None or thresholds is None:
                rospy.logwarn_throttle(5, "Torque data or thresholds None.")
                rate.sleep(); continue

            # Check threshold breach
            if any(abs(tq) > th for tq, th in zip(current_torques, thresholds)):
                collision_detected_in_loop = True
                if not self.collision_event.is_set(): # Capture context on first detection
                     self.collision_torques = list(current_torques) # Store copy
                     self.joint_angles_on_collision = self._get_current_joint_values()
                     rospy.logwarn("Collision TORQUE threshold exceeded!")

            # Set event and stop robot AFTER checking thresholds
            if collision_detected_in_loop and not self.collision_event.is_set():
                rospy.logwarn("Signaling collision event and stopping robot.")
                self.collision_event.set()
                try:
                     self.move_group.stop_robot()
                except Exception as stop_err:
                     rospy.logerr(f"Error stopping robot on collision: {stop_err}")

            try: rate.sleep()
            except rospy.ROSInterruptException: break
        rospy.loginfo("Collision check loop finished.")


    def get_current_ee_pose_stamped(self):
        """Gets the current pose of the end-effector using the interface, returned as PoseStamped."""
        current_pose = self.move_group.get_ee_pose() # Returns geometry_msgs.msg.Pose
        if current_pose is None:
             rospy.logerr("Interface get_ee_pose() returned None.")
             return None

        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = self.planning_frame # Use the actual planning frame
        ps.pose = current_pose
        return ps

    def move_eef_relative(self, dx=0.0, dy=0.0, dz=0.0, description="relative move"):
        """Moves the EEF relatively using Cartesian planning. Returns True on success."""
        current_pose_stamped = self.get_current_ee_pose_stamped()
        if not current_pose_stamped:
            rospy.logerr(f"Cannot get current pose for {description}. Aborting move.")
            return False

        target_pose = Pose()
        target_pose.position.x = current_pose_stamped.pose.position.x + dx
        target_pose.position.y = current_pose_stamped.pose.position.y + dy
        target_pose.position.z = current_pose_stamped.pose.position.z + dz
        target_pose.orientation = current_pose_stamped.pose.orientation # Keep orientation

        rospy.loginfo(f"Planning {description}: dX={dx:.3f}, dY={dy:.3f}, dZ={dz:.3f}")
        # Use a slightly slower speed for recovery/relative moves
        self.move_to_pose(target_pose, description, velocity_scale=0.01, acceleration_scale=0.01)
        rospy.loginfo(f"{description} move completed.")
        return True

    def move_eef_up(self, distance_z=RECOVERY_DISTANCE_Z_M):
        rospy.loginfo(f"Attempting to move EEF straight up by {distance_z} m via interface.")
        q_current = self.move_group.get_current_state()
        current_pose = self.robot.fkine(q_current)
        pose_up = SE3.Tz(RECOVERY_DISTANCE_Z_M) * current_pose
        ik_sol = self.robot.ikine_LM(pose_up, ilimit=40, slimit=500,
                                     joint_limits=True, tol=0.003)
        self.move_to_joint_state(ik_sol.q,"moving up")


    def move_back_after_collision(self, collision_pose, segment_start_pose, segment_end_pose):
        """Moves the EEF back slightly after a collision during Y-scan."""
        rospy.loginfo("Attempting to move back slightly...")

        # Determine direction of the current segment's Y movement
        delta_y_segment = segment_end_pose.position.y - segment_start_pose.position.y

        # Move back opposite to the segment's Y direction
        # If delta_y > 0 (moving towards +Y), move back towards -Y
        # If delta_y < 0 (moving towards -Y), move back towards +Y
        move_back_dy = -math.copysign(RECOVERY_DISTANCE_Y_M, delta_y_segment)

        rospy.loginfo(f"Segment Y delta: {delta_y_segment:.3f}, Moving back by dY: {move_back_dy:.3f}")

        # Use the relative move function for simplicity and safety
        self.move_eef_relative(dy=move_back_dy, description="move back recovery")
        rospy.sleep(0.2) # Pause after moving back
        return True


    def collision_analysis(self,side):
        if self.collision_torques is not None and self.joint_angles_on_collision is not None:
            try:
                rospy.loginfo("Isolating collision source...")
                collision_torques_iso, collided_link_initial = self.isolate_collision_link(self.collision_torques)
                initial_s = self.initial_condition_selection(collided_link_initial)
                collided_link_opt, contact_point_opt, force_estimated_opt, best_s_opt, force_amplitude = self.isolate_collision_point_optimization(
                    collided_link_initial, initial_s, self.joint_angles_on_collision, collision_torques_iso)
                fault_isolation = abs(collided_link_opt - collided_link_initial) > 1
                rospy.loginfo(f"Collision analysis: Initial Link={collided_link_initial}, Opt Link={collided_link_opt}")
                self.visualize_contact_estimation(collided_link_opt, contact_point_opt, force_estimated_opt,
                                                  fault_isolation,side)
            except Exception as analysis_e:
                rospy.logerr(f"Error during collision analysis: {analysis_e}")
        else:
            rospy.logwarn("Collision context not available for analysis.")


    def _generate_waypoints(self, start_pose, end_pose, step_distance):
        """Generates intermediate waypoints between start and end poses."""
        waypoints = []
        start_point = np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z])
        end_point = np.array([end_pose.position.x, end_pose.position.y, end_pose.position.z])
        total_dist = np.linalg.norm(end_point - start_point)

        if total_dist < 1e-4: # If start and end are basically the same
             waypoints.append(start_pose)
             return waypoints

        num_steps = int(math.ceil(total_dist / step_distance))
        if num_steps == 0: num_steps = 1 # Ensure at least one step towards end point

        waypoints.append(start_pose) # Include the starting pose implicitly (robot is already there)
        for i in range(1, num_steps + 1):
            fraction = float(i) / num_steps
            intermediate_point = start_point + fraction * (end_point - start_point)
            wp = Pose()
            wp.position.x = intermediate_point[0]
            wp.position.y = intermediate_point[1]
            wp.position.z = intermediate_point[2] # Keep original Z for now
            wp.orientation = start_pose.orientation # Maintain orientation
            waypoints.append(wp)

        rospy.loginfo(f"Generated {len(waypoints)-1} waypoints for segment (dist: {total_dist:.3f}m).")
        return waypoints[1:] # Return only the target waypoints (excluding the start)


    def perform_planar_scan(self):
        rospy.loginfo("Starting planar scan sequence (waypoint-based)...")

        # Move to a defined starting joint configuration known to be collision-free
        # Example starting config (adjust for your robot)
        scan_start_configuration = [-0.4673, 0.8559, -0.1393, -1.5663, 0.0656, 0.8555, 0.1171]
        rospy.loginfo("Moving to safe scan starting configuration...")
        if not self.move_to_joint_state(scan_start_configuration, description="scan start config"):
            rospy.logerr("Failed to reach starting config. Aborting scan.")
            self.stop_event.set()
        rospy.loginfo("Reached scan starting configuration.")
        rospy.sleep(1.0)

        # Capture current EE pose to get orientation and initial position
        initial_pose_stamped = self.get_current_ee_pose_stamped()
        if not initial_pose_stamped:
            rospy.logerr("Could not get initial EE pose after moving to start config. Aborting.")
            self.stop_event.set(); return
        orientation_to_maintain = initial_pose_stamped.pose.orientation
        rospy.loginfo(f"Maintaining orientation: Q(x={orientation_to_maintain.x:.3f}, y={orientation_to_maintain.y:.3f}, z={orientation_to_maintain.z:.3f}, w={orientation_to_maintain.w:.3f})")

        # --- Generate Scan Segments (Pairs of Start/End Poses for each X-slice) ---
        scan_segments = []
        x_coords = np.arange(self.search_x_min, self.search_x_max + self.search_step_size / 2.0, self.search_step_size) # Include max
        scan_y_left_to_right = True

        for x in x_coords:
            y_start = self.search_y_min if scan_y_left_to_right else self.search_y_max
            y_end = self.search_y_max if scan_y_left_to_right else self.search_y_min

            start_pose = Pose()
            start_pose.position.x = x
            start_pose.position.y = y_start
            start_pose.position.z = self.search_z_height
            start_pose.orientation = orientation_to_maintain

            end_pose = Pose()
            end_pose.position.x = x
            end_pose.position.y = y_end
            end_pose.position.z = self.search_z_height
            end_pose.orientation = orientation_to_maintain

            scan_segments.append((start_pose, end_pose))
            scan_y_left_to_right = not scan_y_left_to_right
        # --- End Segment Generation ---

        if not scan_segments:
            rospy.logerr("No scan segments generated. Check parameters."); self.stop_event.set(); return
        rospy.loginfo(f"Generated {len(scan_segments)} scan segments.")

        # --- Execute Segments ---
        total_segments = len(scan_segments)
        for i, (segment_start, segment_end) in enumerate(scan_segments):
            segment_desc = f"Segment {i+1}/{total_segments} (X={segment_start.position.x:.3f})"
            rospy.loginfo(f"--- Starting {segment_desc} ---")
            self.current_segment_start_pose = segment_start
            self.current_segment_end_pose = segment_end

            # 1. Move to the start pose of the current segment
            rospy.loginfo(f"Moving to segment start: Y={segment_start.position.y:.3f}, Z={segment_start.position.z:.3f}")
            if not self.move_to_pose(segment_start, f"start of {segment_desc}", velocity_scale=0.15, acceleration_scale=0.15):
                if self.collision_event.is_set():
                     rospy.logwarn(f"Collision moving to start of {segment_desc}. Attempting generic recovery.")
                     rospy.logerr("Collision recovery failed. Aborting scan.")
                     self.stop_event.set()
                else:
                     rospy.logerr(f"Failed to reach start of {segment_desc} (no collision). Aborting scan.")
                     self.stop_event.set(); break # Abort if start pose unreachable

            if self.stop_event.is_set() or rospy.is_shutdown(): break
            rospy.loginfo(f"Reached start of {segment_desc}.")
            rospy.sleep(0.2)

            # 2. Generate waypoints for this segment
            waypoints = self._generate_waypoints(segment_start, segment_end, WAYPOINT_DISTANCE_M)
            if not waypoints:
                rospy.logwarn(f"No waypoints generated for {segment_desc}, skipping to next.")
                continue

            # 3. Execute move through waypoints with recovery logic
            is_raised = False
            current_z_offset = 0.0
            raise_retries = 0
            segment_failed = False

            num_waypoints = len(waypoints)
            for wp_idx, target_waypoint in enumerate(waypoints):
                waypoint_desc = f"Waypoint {wp_idx+1}/{num_waypoints} in {segment_desc}"

                if self.stop_event.is_set() or rospy.is_shutdown(): segment_failed = True; break

                # Adjust target Z if we are in raised state
                effective_target_pose = Pose()
                effective_target_pose.position.x = target_waypoint.position.x
                effective_target_pose.position.y = target_waypoint.position.y
                effective_target_pose.position.z = target_waypoint.position.z + current_z_offset # Apply offset if raised
                effective_target_pose.orientation = target_waypoint.orientation

                rospy.loginfo(f"Moving to {waypoint_desc} {'(Raised Z)' if is_raised else ''}")
                self.collision_event.clear() # Clear before attempting move

                # --- Attempt Move to Waypoint ---
                #move_success = self.move_to_pose(effective_target_pose, waypoint_desc)
                move_success = False #remove if not useing cartesian
                (plan, fraction) = self.move_group.plan_cartesian_path([effective_target_pose], collisions=True)
                min_required_fraction = 0.9
                if fraction < min_required_fraction:
                    rospy.logerr(f"Planning failed for {segment_desc} ({fraction * 100:.1f}%). Aborting.");
                    self.stop_event.set()
                    break
                self.move_group.execute_plan(plan, wait=True)
                # --- Handle Outcome ---
                if self.stop_event.is_set() or rospy.is_shutdown():
                    segment_failed = True
                    break

                if self.collision_event.is_set():
                    # *** COLLISION DETECTED ***
                    rospy.logwarn(f"Collision detected moving to {waypoint_desc}.")
                    collision_pose_stamped = self.get_current_ee_pose_stamped()
                    collision_pose = collision_pose_stamped.pose if collision_pose_stamped else None

                    if collision_pose is None:
                         rospy.logerr("Could not get collision pose. Cannot recover. Aborting segment.")
                         segment_failed = True; break

                    # 1. Analyze Collision
                    self.collision_analysis(i) # Use stored context

                    # 2. Move Back
                    if not self.move_back_after_collision(collision_pose, segment_start, segment_end):
                         rospy.logerr("Failed to move back during recovery. Aborting segment.")
                         segment_failed = True; break
                    if self.stop_event.is_set() or rospy.is_shutdown():
                        segment_failed = True
                        break
                    rospy.sleep(0.1)
                    self.collision_event.clear() # Clear event after moving back

                    # 3. Move Up
                    rospy.loginfo(f"Moving EEF up by {RECOVERY_DISTANCE_Z_M}m (Attempt {raise_retries+1})")
                    #self.move_eef_up(distance_z=RECOVERY_DISTANCE_Z_M)
                    self.move_eef_relative(dz=RECOVERY_DISTANCE_Z_M, description="raise recovery")
                    if self.stop_event.is_set() or rospy.is_shutdown():
                        segment_failed = True
                        break

                    current_z_offset += RECOVERY_DISTANCE_Z_M
                    is_raised = True # Enter raised state
                    raise_retries += 1

                    # Clear context used by analysis AFTER attempting recovery moves
                    self.collision_torques = None
                    self.joint_angles_on_collision = None
                    self.collision_event.clear() # Ensure clear before retrying waypoint

                    # Check retry limit
                    if raise_retries > MAX_RAISE_RETRIES:
                        rospy.logerr(f"Max raise retries ({MAX_RAISE_RETRIES}) exceeded for {waypoint_desc}. Aborting segment.")
                        segment_failed = True; break

                    rospy.loginfo(f"Recovery attempted. Retrying {waypoint_desc} at Z + {current_z_offset:.3f}m.")
                    rospy.sleep(0.1)
                    continue # Skip to next iteration of waypoint loop (retry same wp_idx)

                rospy.loginfo(f"Successfully reached {waypoint_desc}.")
                raise_retries = 0 # Reset retry counter upon successful move to a waypoint

                if is_raised and (wp_idx +1) == num_waypoints:
                    # If we are raised, attempt to descend AT this waypoint
                    rospy.loginfo(f"Currently raised (Z offset: {current_z_offset:.3f}m). Attempting descent at {waypoint_desc}.")
                    descend_target_pose = Pose()
                    descend_target_pose.position.x = target_waypoint.position.x
                    descend_target_pose.position.y = target_waypoint.position.y
                    descend_target_pose.position.z = self.search_z_height # Target original Z
                    descend_target_pose.orientation = target_waypoint.orientation

                    self.collision_event.clear()
                    # Use slower speed for cautious descent
                    self.move_to_pose(descend_target_pose, f"descend attempt at {waypoint_desc}", velocity_scale=0.003, acceleration_scale=0.003)

                    if self.collision_event.is_set():
                         # Collision during descent
                         rospy.logwarn(f"Collision detected while attempting to descend at {waypoint_desc}. Staying raised.")
                         # Move back up slightly to ensure clearance after collision stop
                         self.collision_event.clear()
                         rospy.sleep(0.1)
                         self.move_eef_relative(dz=0.05, description="minor lift after descend collision") # Small lift
                         self.collision_event.clear()
                         # Keep is_raised = True, keep current_z_offset. Proceed to next waypoint raised.
                         rospy.loginfo(f"Will proceed to next waypoint while raised.")

                    else:
                         # Descent successful!
                         rospy.loginfo(f"Successfully descended to original Z height at {waypoint_desc}.")
                         is_raised = False
                         current_z_offset = 0.0
                         # Proceed to next waypoint at normal height

                # Pause briefly between waypoints
                # End of waypoint loop iteration

            if segment_failed:
                rospy.logerr(f"{segment_desc} FAILED. Aborting remaining scan.")
                self.stop_event.set() # Signal stop to other threads
                break # Exit segment loop

            rospy.loginfo(f"--- Completed {segment_desc} ---")
            rospy.sleep(0.2)

        # --- Scan Finished or Aborted ---
        if self.stop_event.is_set():
            rospy.logwarn("Planar scan sequence aborted.")
        elif rospy.is_shutdown():
            rospy.loginfo("Planar scan sequence interrupted by ROS shutdown.")
        else:
            rospy.loginfo("Planar scan sequence finished.")

        # --- Move Home Safely ---
        rospy.loginfo("Moving to safe 'home' position.")
        if "home" in self.position_states:
             # Use slower speed for final homing
             self.move_to_joint_state(self.position_states["home"], description="final home")
        else:
            rospy.logwarn("Home position not defined in position_states.")
        rospy.loginfo("Reached home position.")



    def start(self):
        if self.move_thread and self.move_thread.is_alive() or self.collision_check_thread and self.collision_check_thread.is_alive():
            rospy.logwarn("Threads running. Call shutdown() first.")
            return
        rospy.loginfo("Initializing and starting threads...")
        self.stop_event.clear()
        self.collision_event.clear()
        self.collision_torques = None # Clear context on start
        self.joint_angles_on_collision = None
        self.contact_point_log_counter = 0
        self.collision_check_thread = threading.Thread(target=self.check_collision_loop, daemon=True);
        self.collision_check_thread.start()
        rospy.loginfo("Collision check thread started.")
        rospy.sleep(0.5) # Give collision check time to initialize
        self.move_thread = threading.Thread(target=self.perform_planar_scan);
        self.move_thread.start()
        rospy.loginfo("Planar scan sequence thread started.")

    def shutdown(self):
        if self.stop_event.is_set(): rospy.loginfo("Shutdown already in progress."); return
        rospy.loginfo("Initiating shutdown sequence...");
        self.stop_event.set()
        if self.move_group:
            rospy.loginfo("Stopping robot motion via Interface...")
            try:
                self.move_group.stop_robot()
            except Exception as e:
                rospy.logerr(f"Error stopping move_group: {e}")
        if self.collision_check_thread and self.collision_check_thread.is_alive():
            rospy.loginfo("Waiting for collision check thread...");
            self.collision_check_thread.join(timeout=2.0)
            if self.collision_check_thread.is_alive(): rospy.logwarn("Collision check thread did not stop.")
        if self.move_thread and self.move_thread.is_alive():
            rospy.loginfo("Waiting for movement thread...");
            self.move_thread.join(timeout=5.0) # Allow more time as it might be in a move
            if self.move_thread.is_alive(): rospy.logwarn("Movement thread did not stop.")
        rospy.loginfo("Exploration node shutdown complete.")

if __name__ == "__main__":
    node_name = 'explore_table_plane_node'
    try:
        rospy.init_node(node_name, anonymous=True)
        rospy.loginfo(f"'{node_name}' Node Started")


        threshold_values = [5.0, 5.0, 4.0, 4.0, 3.0, 2.0, 1.0]
        rospy.loginfo(f"Using torque thresholds: {threshold_values}")
        SCAN_X_MIN = 0.4
        SCAN_X_MAX = 0.7
        SCAN_Y_MIN = -0.32
        SCAN_Y_MAX = 0.32
        SCAN_Z_HEIGHT = 0.075  # Target Z for normal operation
        SCAN_STEP_X = 0.05  # Step size along X

        try:
            collision_detector = CollisionDetection(thresholds=threshold_values, buffer_size=10)
            rospy.loginfo("CollisionDetection instance created successfully.")
            rospy.sleep(1.5) # Allow time for subscriptions etc.
        except Exception as e:
            rospy.logfatal(f"Failed to initialize CollisionDetection: {e}");
            import traceback; traceback.print_exc(); exit()

        # Instantiate ExploreTablePlane
        explore_table = ExploreTablePlane(
            collision_detection=collision_detector,
            search_x_min=SCAN_X_MIN,
            search_x_max=SCAN_X_MAX,
            search_y_min=SCAN_Y_MIN,
            search_y_max=SCAN_Y_MAX,
            search_z_height=SCAN_Z_HEIGHT,
            search_step_size=SCAN_STEP_X
        )

        def shutdown_hook():
            rospy.logwarn(f"'{node_name}' received shutdown signal.")
            explore_table.shutdown()

        rospy.on_shutdown(shutdown_hook)

        explore_table.start()
        rospy.loginfo(f"'{node_name}' spinning. Press Ctrl+C to stop.")
        rospy.spin() # Keep node alive until shutdown

    except rospy.ROSInterruptException:
        rospy.loginfo(f"'{node_name}' interrupted.")
    except Exception as e:
        rospy.logfatal(f"An unhandled error occurred in '{node_name}': {e}");
        import traceback; traceback.print_exc()
    finally:
        # Ensure shutdown is called even if errors occur during spin/start
        if 'explore_table' in locals() and explore_table:
            rospy.loginfo("Ensuring shutdown in finally block...")
            explore_table.shutdown()
        rospy.loginfo(f"'{node_name}' exiting.")