#!/usr/bin/env python3

import math
import os
from enum import Enum
from typing import List, Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_planning_msgs.msg import Trajectory
from autoware_auto_vehicle_msgs.msg import GearCommand
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from v2x_msgs.msg import V2XVehiclePositionArray

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    forward_waypoint_distance,
    nearest_waypoint_index,
)


class RecoveryState(Enum):
    IDLE = "idle"
    BRAKE = "brake"
    SHIFT_REVERSE = "shift_reverse"
    REVERSE = "reverse"
    SHIFT_DRIVE = "shift_drive"
    FORWARD = "forward"
    COOLDOWN = "cooldown"


class ControlCmdMux(Node):
    def __init__(self) -> None:
        super().__init__("control_cmd_mux")

        self.declare_parameter("pure_pursuit_cmd_topic", "/control/command/pure_pursuit_cmd")
        self.declare_parameter("mpc_cmd_topic", "/control/command/mpc_cmd")
        self.declare_parameter("output_cmd_topic", "/control/command/control_cmd")
        self.declare_parameter("gear_cmd_topic", "/control/command/gear_cmd")
        self.declare_parameter("kinematics_topic", "/localization/kinematic_state")
        self.declare_parameter("trajectory_topic", "/planning/scenario_planning/trajectory")
        self.declare_parameter("v2x_topic", "/v2x/vehicle_positions")
        self.declare_parameter("ego_vehicle_id", os.environ.get("VEHICLE_ID", ""))
        self.declare_parameter("enabled_ego_vehicle_id", "P1")
        self.declare_parameter("switch_waypoint_count", 6)
        self.declare_parameter("release_waypoint_count", 1)
        self.declare_parameter("v2x_stale_timeout", 0.5)
        self.declare_parameter("enable_recovery", True)
        self.declare_parameter("condition_topic", "/aichallenge/pitstop/condition")
        self.declare_parameter("condition_jump_threshold", 30.0)
        self.declare_parameter("stuck_speed_threshold", 0.2)
        self.declare_parameter("stuck_command_speed_threshold", 1.0)
        self.declare_parameter("stuck_duration", 1.0)
        self.declare_parameter("recovery_startup_grace", 8.0)
        self.declare_parameter("recovery_enable_after_speed", 0.5)
        self.declare_parameter("recovery_brake_duration", 0.3)
        self.declare_parameter("recovery_shift_reverse_duration", 0.5)
        self.declare_parameter("recovery_min_reverse_duration", 0.5)
        self.declare_parameter("recovery_max_reverse_duration", 2.5)
        self.declare_parameter("recovery_min_reverse_distance", 0.8)
        self.declare_parameter("recovery_shift_drive_duration", 0.3)
        self.declare_parameter("recovery_min_forward_duration", 0.5)
        self.declare_parameter("recovery_max_forward_duration", 0.0)
        self.declare_parameter("recovery_blocked_speed_threshold", 0.1)
        self.declare_parameter("recovery_blocked_duration", 0.8)
        self.declare_parameter("recovery_use_nan_speed", False)
        self.declare_parameter("recovery_speed", 1.0)
        self.declare_parameter("recovery_forward_speed", 1.2)
        self.declare_parameter("recovery_acceleration", 3.0)
        self.declare_parameter("recovery_steering_mode", "path_approach")
        self.declare_parameter("recovery_max_steering_angle", 0.5)
        self.declare_parameter("recovery_path_distance_threshold", 0.8)
        self.declare_parameter("recovery_path_approach_distance", 2.0)
        self.declare_parameter("recovery_cooldown", 1.0)

        pp_topic = str(self.get_parameter("pure_pursuit_cmd_topic").value)
        mpc_topic = str(self.get_parameter("mpc_cmd_topic").value)
        output_topic = str(self.get_parameter("output_cmd_topic").value)
        gear_cmd_topic = str(self.get_parameter("gear_cmd_topic").value)
        kinematics_topic = str(self.get_parameter("kinematics_topic").value)
        trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        v2x_topic = str(self.get_parameter("v2x_topic").value)
        condition_topic = str(self.get_parameter("condition_topic").value)

        self._ego_vehicle_id = str(self.get_parameter("ego_vehicle_id").value)
        self._enabled_ego_vehicle_id = str(self.get_parameter("enabled_ego_vehicle_id").value)
        self._switch_waypoint_count = int(self.get_parameter("switch_waypoint_count").value)
        self._release_waypoint_count = int(self.get_parameter("release_waypoint_count").value)
        self._v2x_stale_timeout = float(self.get_parameter("v2x_stale_timeout").value)
        self._enable_recovery = bool(self.get_parameter("enable_recovery").value)
        self._condition_jump_threshold = float(self.get_parameter("condition_jump_threshold").value)
        self._stuck_speed_threshold = float(self.get_parameter("stuck_speed_threshold").value)
        self._stuck_command_speed_threshold = float(
            self.get_parameter("stuck_command_speed_threshold").value)
        self._stuck_duration = float(self.get_parameter("stuck_duration").value)
        self._recovery_startup_grace = float(self.get_parameter("recovery_startup_grace").value)
        self._recovery_enable_after_speed = float(
            self.get_parameter("recovery_enable_after_speed").value)
        self._recovery_brake_duration = float(self.get_parameter("recovery_brake_duration").value)
        self._recovery_shift_reverse_duration = float(
            self.get_parameter("recovery_shift_reverse_duration").value)
        self._recovery_min_reverse_duration = float(
            self.get_parameter("recovery_min_reverse_duration").value)
        self._recovery_max_reverse_duration = float(
            self.get_parameter("recovery_max_reverse_duration").value)
        self._recovery_min_reverse_distance = float(
            self.get_parameter("recovery_min_reverse_distance").value)
        self._recovery_shift_drive_duration = float(
            self.get_parameter("recovery_shift_drive_duration").value)
        self._recovery_min_forward_duration = float(
            self.get_parameter("recovery_min_forward_duration").value)
        self._recovery_max_forward_duration = float(
            self.get_parameter("recovery_max_forward_duration").value)
        self._recovery_blocked_speed_threshold = float(
            self.get_parameter("recovery_blocked_speed_threshold").value)
        self._recovery_blocked_duration = float(
            self.get_parameter("recovery_blocked_duration").value)
        self._recovery_use_nan_speed = bool(self.get_parameter("recovery_use_nan_speed").value)
        self._recovery_speed = float(self.get_parameter("recovery_speed").value)
        self._recovery_forward_speed = float(self.get_parameter("recovery_forward_speed").value)
        self._recovery_acceleration = float(self.get_parameter("recovery_acceleration").value)
        self._recovery_steering_mode = str(self.get_parameter("recovery_steering_mode").value)
        self._recovery_max_steering_angle = float(
            self.get_parameter("recovery_max_steering_angle").value)
        self._recovery_path_distance_threshold = float(
            self.get_parameter("recovery_path_distance_threshold").value)
        self._recovery_path_approach_distance = float(
            self.get_parameter("recovery_path_approach_distance").value)
        self._recovery_cooldown = float(self.get_parameter("recovery_cooldown").value)

        self._odom: Optional[Odometry] = None
        self._trajectory: Optional[Trajectory] = None
        self._vehicles: List[Tuple[str, float, float]] = []
        self._last_v2x_time = None
        self._use_mpc = False
        self._mpc_target_vehicle_id: Optional[str] = None
        self._last_pp_cmd: Optional[AckermannControlCommand] = None
        self._last_mpc_cmd: Optional[AckermannControlCommand] = None
        self._stuck_since = None
        self._recovery_state = RecoveryState.IDLE
        self._node_start_time = self.get_clock().now()
        self._has_moved = False
        self._recovery_brake_until = None
        self._recovery_shift_reverse_until = None
        self._recovery_reverse_started_at = None
        self._recovery_reverse_start_pose: Optional[Tuple[float, float]] = None
        self._recovery_reverse_until = None
        self._recovery_shift_drive_until = None
        self._recovery_forward_started_at = None
        self._recovery_forward_until = None
        self._recovery_blocked_since = None
        self._recovery_cooldown_until = None
        self._last_condition: Optional[int] = None

        self._pub = self.create_publisher(AckermannControlCommand, output_topic, 1)
        self._gear_pub = self.create_publisher(GearCommand, gear_cmd_topic, 1)
        self.create_subscription(AckermannControlCommand, pp_topic, self._pp_cmd_callback, 1)
        self.create_subscription(AckermannControlCommand, mpc_topic, self._mpc_cmd_callback, 1)
        self.create_subscription(Odometry, kinematics_topic, self._odom_callback, 1)
        trajectory_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Trajectory, trajectory_topic, self._trajectory_callback, trajectory_qos)
        self.create_subscription(V2XVehiclePositionArray, v2x_topic, self._v2x_callback, 1)
        self.create_subscription(Int32, condition_topic, self._condition_callback, 1)
        self.create_timer(0.1, self._on_timer)

        self.get_logger().info(
            f"Control mux started: PP='{pp_topic}', MPC='{mpc_topic}', output='{output_topic}', "
            f"switch when vehicle ahead <= {self._switch_waypoint_count} wp, "
            f"release after ego leads by >= {self._release_waypoint_count} wp")

    def _pp_cmd_callback(self, msg: AckermannControlCommand) -> None:
        self._last_pp_cmd = msg
        if not self._use_mpc and not self._is_recovering():
            self._pub.publish(msg)

    def _mpc_cmd_callback(self, msg: AckermannControlCommand) -> None:
        self._last_mpc_cmd = msg
        if self._use_mpc and not self._is_recovering():
            self._pub.publish(msg)

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg
        self._update_stuck_detection()
        self._update_mode()

    def _trajectory_callback(self, msg: Trajectory) -> None:
        self._trajectory = msg

    def _v2x_callback(self, msg: V2XVehiclePositionArray) -> None:
        vehicles = []
        for vehicle in msg.vehicles:
            if self._ego_vehicle_id and vehicle.vehicle_id == self._ego_vehicle_id:
                continue
            vehicles.append((
                vehicle.vehicle_id,
                float(vehicle.position.x),
                float(vehicle.position.y),
            ))
        self._vehicles = vehicles
        self._last_v2x_time = self.get_clock().now()
        self._update_mode()

    def _condition_callback(self, msg: Int32) -> None:
        if self._last_condition is None:
            self._last_condition = msg.data
            return
        if self._has_moved and msg.data - self._last_condition > self._condition_jump_threshold:
            self._start_recovery("condition jump indicates collision")
        self._last_condition = msg.data

    def _on_timer(self) -> None:
        if self._recovery_state in (
            RecoveryState.BRAKE,
            RecoveryState.SHIFT_REVERSE,
            RecoveryState.REVERSE,
            RecoveryState.SHIFT_DRIVE,
            RecoveryState.FORWARD,
        ):
            self._publish_recovery_command()
            return
        if self._recovery_state == RecoveryState.COOLDOWN:
            if self._recovery_cooldown_until is not None and self.get_clock().now() < self._recovery_cooldown_until:
                return
            self._recovery_state = RecoveryState.IDLE
            self.get_logger().info("Recovery finished; resuming selected controller")
            self._publish_gear(GearCommand.DRIVE)
            cmd = self._last_mpc_cmd if self._use_mpc else self._last_pp_cmd
            if cmd is not None:
                self._pub.publish(cmd)
        self._stale_check()

    def _stale_check(self) -> None:
        if self._last_v2x_time is None:
            return
        age = (self.get_clock().now() - self._last_v2x_time).nanoseconds * 1e-9
        if age > self._v2x_stale_timeout:
            self._vehicles = []
            self._mpc_target_vehicle_id = None
            self._set_mode(False, "V2X timeout")

    def _update_stuck_detection(self) -> None:
        if not self._enable_recovery or self._odom is None or self._recovery_state != RecoveryState.IDLE:
            return

        selected_cmd = self._last_mpc_cmd if self._use_mpc else self._last_pp_cmd
        if selected_cmd is None:
            self._stuck_since = None
            return

        current_speed = abs(float(self._odom.twist.twist.linear.x))
        if current_speed >= self._recovery_enable_after_speed:
            self._has_moved = True

        now = self.get_clock().now()
        elapsed_from_start = (now - self._node_start_time).nanoseconds * 1e-9
        if not self._has_moved:
            self._stuck_since = None
            return
        if elapsed_from_start < self._recovery_startup_grace:
            self._stuck_since = None
            return

        command_speed = float(selected_cmd.longitudinal.speed)
        trying_to_move_forward = command_speed > self._stuck_command_speed_threshold
        is_nearly_stopped = current_speed < self._stuck_speed_threshold
        if trying_to_move_forward and is_nearly_stopped:
            if self._stuck_since is None:
                self._stuck_since = now
                return
            elapsed = (now - self._stuck_since).nanoseconds * 1e-9
            if elapsed >= self._stuck_duration:
                self._start_recovery(
                    f"stuck: speed={current_speed:.2f} m/s, cmd={command_speed:.2f} m/s")
        else:
            self._stuck_since = None

    def _is_recovering(self) -> bool:
        return self._recovery_state in (
            RecoveryState.BRAKE,
            RecoveryState.SHIFT_REVERSE,
            RecoveryState.REVERSE,
            RecoveryState.SHIFT_DRIVE,
            RecoveryState.FORWARD,
        )

    def _start_recovery(self, reason: str) -> None:
        if not self._enable_recovery or self._is_recovering():
            return
        now = self.get_clock().now()
        self._recovery_state = RecoveryState.BRAKE
        self._recovery_brake_until = now + Duration(seconds=self._recovery_brake_duration)
        self._recovery_shift_reverse_until = (
            self._recovery_brake_until + Duration(seconds=self._recovery_shift_reverse_duration)
        )
        if self._recovery_max_reverse_duration > 0.0:
            self._recovery_reverse_until = (
                self._recovery_shift_reverse_until
                + Duration(seconds=self._recovery_max_reverse_duration)
            )
        else:
            self._recovery_reverse_until = None
        self._recovery_reverse_started_at = None
        self._recovery_reverse_start_pose = None
        self._recovery_shift_drive_until = None
        self._recovery_forward_started_at = None
        self._recovery_forward_until = None
        self._recovery_blocked_since = None
        self._stuck_since = None
        self.get_logger().warn(
            f"Starting recovery: brake {self._recovery_brake_duration:.1f}s, "
            f"shift reverse {self._recovery_shift_reverse_duration:.1f}s, "
            f"reverse at least {self._recovery_min_reverse_distance:.1f}m, "
            f"short reverse then path-approach forward if needed: {reason}")
        self._publish_recovery_command()

    def _publish_recovery_command(self) -> None:
        now = self.get_clock().now()
        if self._recovery_state == RecoveryState.BRAKE:
            if self._recovery_brake_until is not None and now >= self._recovery_brake_until:
                self._recovery_state = RecoveryState.SHIFT_REVERSE
                self._publish_gear(GearCommand.REVERSE)
            else:
                self._publish_gear(GearCommand.DRIVE)
                self._publish_manual_command(speed=0.0, acceleration=-3.0, steering=0.0)
                return

        if self._recovery_state == RecoveryState.SHIFT_REVERSE:
            if (
                self._recovery_shift_reverse_until is not None
                and now >= self._recovery_shift_reverse_until
            ):
                self._recovery_state = RecoveryState.REVERSE
                self._recovery_reverse_started_at = now
                self._recovery_reverse_start_pose = self._current_xy()
            else:
                self._publish_gear(GearCommand.REVERSE)
                self._publish_manual_command(speed=0.0, acceleration=0.0, steering=0.0)
                return

        if self._recovery_state == RecoveryState.REVERSE:
            should_finish = self._should_finish_reverse_recovery(now)
            timed_out = (
                self._recovery_reverse_until is not None
                and now >= self._recovery_reverse_until
            )
            rear_blocked = self._is_recovery_direction_blocked(now, self._recovery_reverse_started_at)
            if should_finish or timed_out:
                if timed_out and not should_finish:
                    self._start_drive_recovery(now, "reverse did not reach path")
                    return
                self._recovery_state = RecoveryState.COOLDOWN
                self._recovery_cooldown_until = now + Duration(seconds=self._recovery_cooldown)
                self._publish_gear(GearCommand.DRIVE)
                self._publish_manual_command(speed=0.0, acceleration=0.0, steering=0.0)
                return
            if rear_blocked:
                self._start_drive_recovery(now, "reverse appears blocked")
                return
            self._publish_gear(GearCommand.REVERSE)
            self._publish_manual_command(
                speed=float("nan") if self._recovery_use_nan_speed else self._recovery_speed,
                acceleration=self._recovery_acceleration,
                steering=self._recovery_steering_angle(reverse=True))

        if self._recovery_state == RecoveryState.SHIFT_DRIVE:
            if self._recovery_shift_drive_until is not None and now >= self._recovery_shift_drive_until:
                self._recovery_state = RecoveryState.FORWARD
                self._recovery_forward_started_at = now
                if self._recovery_max_forward_duration > 0.0:
                    self._recovery_forward_until = now + Duration(
                        seconds=self._recovery_max_forward_duration)
                else:
                    self._recovery_forward_until = None
            else:
                self._publish_gear(GearCommand.DRIVE)
                self._publish_manual_command(speed=0.0, acceleration=0.0, steering=0.0)
                return

        if self._recovery_state == RecoveryState.FORWARD:
            should_finish = self._should_finish_forward_recovery(now)
            timed_out = (
                self._recovery_forward_until is not None
                and now >= self._recovery_forward_until
            )
            if should_finish or timed_out:
                self._recovery_state = RecoveryState.COOLDOWN
                self._recovery_cooldown_until = now + Duration(seconds=self._recovery_cooldown)
                self._publish_gear(GearCommand.DRIVE)
                self._publish_manual_command(speed=0.0, acceleration=0.0, steering=0.0)
                if timed_out and not should_finish:
                    self.get_logger().warn("Forward recovery timed out before reaching path")
                return
            self._publish_gear(GearCommand.DRIVE)
            self._publish_manual_command(
                speed=self._recovery_forward_speed,
                acceleration=self._recovery_acceleration,
                steering=self._recovery_steering_angle(reverse=False))

    def _start_drive_recovery(self, now, reason: str) -> None:
        self._recovery_state = RecoveryState.SHIFT_DRIVE
        self._recovery_shift_drive_until = now + Duration(seconds=self._recovery_shift_drive_duration)
        self._recovery_forward_started_at = None
        self._recovery_forward_until = None
        self._recovery_blocked_since = None
        self.get_logger().warn(f"Switching recovery to DRIVE: {reason}")
        self._publish_gear(GearCommand.DRIVE)
        self._publish_manual_command(speed=0.0, acceleration=0.0, steering=0.0)

    def _is_recovery_direction_blocked(self, now, started_at) -> bool:
        if self._odom is None or started_at is None:
            return False
        elapsed = (now - started_at).nanoseconds * 1e-9
        if elapsed < self._recovery_blocked_duration:
            return False

        speed = abs(float(self._odom.twist.twist.linear.x))
        if speed >= self._recovery_blocked_speed_threshold:
            self._recovery_blocked_since = None
            return False
        if self._recovery_blocked_since is None:
            self._recovery_blocked_since = now
            return False
        blocked_elapsed = (now - self._recovery_blocked_since).nanoseconds * 1e-9
        return blocked_elapsed >= self._recovery_blocked_duration

    def _publish_gear(self, command: int) -> None:
        gear = GearCommand()
        gear.stamp = self.get_clock().now().to_msg()
        gear.command = command
        self._gear_pub.publish(gear)

    def _publish_manual_command(self, speed: float, acceleration: float, steering: float) -> None:
        cmd = AckermannControlCommand()
        stamp = self.get_clock().now().to_msg()
        cmd.stamp = stamp
        cmd.longitudinal.stamp = stamp
        cmd.longitudinal.speed = speed
        cmd.longitudinal.acceleration = acceleration
        cmd.lateral.stamp = stamp
        cmd.lateral.steering_tire_angle = steering
        cmd.lateral.steering_tire_rotation_rate = 2.0
        self._pub.publish(cmd)

    def _recovery_steering_angle(self, reverse: bool) -> float:
        if self._recovery_steering_mode == "path_approach":
            nearest_path = self._nearest_path_state()
            if nearest_path is not None and self._odom is not None:
                path_yaw, _path_distance, lateral_error = nearest_path
                lateral_heading = math.atan2(
                    lateral_error,
                    max(self._recovery_path_approach_distance, 1e-3))
                target_yaw = path_yaw - lateral_heading
                yaw_error = self._normalize_angle(target_yaw - self._odom_yaw())
                steer_sign = 1.0 if yaw_error >= 0.0 else -1.0
                if reverse:
                    steer_sign *= -1.0
                return steer_sign * self._recovery_max_steering_angle

        selected_cmd = self._last_mpc_cmd if self._use_mpc else self._last_pp_cmd
        if selected_cmd is None:
            return 0.0

        angle = float(selected_cmd.lateral.steering_tire_angle)
        if self._recovery_steering_mode == "hold":
            return angle
        if self._recovery_steering_mode == "invert":
            return -angle
        return 0.0

    def _should_finish_reverse_recovery(self, now) -> bool:
        if self._recovery_reverse_started_at is None:
            return False
        elapsed = (now - self._recovery_reverse_started_at).nanoseconds * 1e-9
        if elapsed < self._recovery_min_reverse_duration:
            return False
        return self._has_reversed_enough() and self._is_close_to_path()

    def _has_reversed_enough(self) -> bool:
        if self._recovery_min_reverse_distance <= 0.0:
            return True
        start_xy = self._recovery_reverse_start_pose
        current_xy = self._current_xy()
        if start_xy is None or current_xy is None:
            return False
        return math.hypot(current_xy[0] - start_xy[0], current_xy[1] - start_xy[1]) >= (
            self._recovery_min_reverse_distance
        )

    def _current_xy(self) -> Optional[Tuple[float, float]]:
        if self._odom is None:
            return None
        return (
            float(self._odom.pose.pose.position.x),
            float(self._odom.pose.pose.position.y),
        )

    def _should_finish_forward_recovery(self, now) -> bool:
        if self._recovery_forward_started_at is None:
            return False
        elapsed = (now - self._recovery_forward_started_at).nanoseconds * 1e-9
        if elapsed < self._recovery_min_forward_duration:
            return False
        return self._is_close_to_path()

    def _is_close_to_path(self) -> bool:
        nearest_path = self._nearest_path_state()
        if nearest_path is None or self._odom is None:
            return False
        _path_yaw, path_distance, _lateral_error = nearest_path
        return path_distance <= self._recovery_path_distance_threshold

    def _nearest_path_state(self) -> Optional[Tuple[float, float, float]]:
        if self._trajectory is None or self._odom is None:
            return None
        points = self._trajectory.points
        if len(points) < 2:
            return None

        x = float(self._odom.pose.pose.position.x)
        y = float(self._odom.pose.pose.position.y)
        best_idx = 0
        best_dist_sq = float("inf")
        for i in range(len(points) - 1):
            x0 = float(points[i].pose.position.x)
            y0 = float(points[i].pose.position.y)
            x1 = float(points[i + 1].pose.position.x)
            y1 = float(points[i + 1].pose.position.y)
            vx = x1 - x0
            vy = y1 - y0
            length_sq = vx * vx + vy * vy
            if length_sq <= 1e-6:
                continue
            t = max(0.0, min(1.0, ((x - x0) * vx + (y - y0) * vy) / length_sq))
            proj_x = x0 + t * vx
            proj_y = y0 + t * vy
            dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = i

        p0 = points[best_idx].pose.position
        p1 = points[best_idx + 1].pose.position
        dx = float(p1.x) - float(p0.x)
        dy = float(p1.y) - float(p0.y)
        path_yaw = math.atan2(dy, dx)
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return path_yaw, math.sqrt(best_dist_sq), 0.0

        # Positive means the vehicle is left of the path direction.
        lateral_error = (dx * (y - float(p0.y)) - dy * (x - float(p0.x))) / length
        return path_yaw, math.sqrt(best_dist_sq), lateral_error

    def _odom_yaw(self) -> float:
        q = self._odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _update_mode(self) -> None:
        if self._enabled_ego_vehicle_id and self._ego_vehicle_id != self._enabled_ego_vehicle_id:
            self._mpc_target_vehicle_id = None
            self._set_mode(False, "costmap overtake is disabled for this ego vehicle")
            return
        if self._odom is None:
            return
        if self._trajectory is None or not self._trajectory.points:
            return

        ego_x = float(self._odom.pose.pose.position.x)
        ego_y = float(self._odom.pose.pose.position.y)
        relations = self._vehicle_waypoint_relations(ego_x, ego_y)
        if not relations:
            self._mpc_target_vehicle_id = None
            self._set_mode(False, "no nearby V2X vehicles")
            return

        switch_candidate = min(
            (
                relation for relation in relations
                if relation[1] <= self._switch_waypoint_count
            ),
            key=lambda relation: relation[1],
            default=None,
        )
        if switch_candidate is not None:
            vehicle_id, vehicle_ahead_distance, _ego_ahead_distance = switch_candidate
            self._mpc_target_vehicle_id = vehicle_id
            self._set_mode(
                True,
                f"vehicle '{vehicle_id}' is {vehicle_ahead_distance} waypoints ahead")
            return

        if self._use_mpc and self._mpc_target_vehicle_id:
            target = next(
                (
                    relation for relation in relations
                    if relation[0] == self._mpc_target_vehicle_id
                ),
                None,
            )
            if target is not None:
                vehicle_id, _vehicle_ahead_distance, ego_ahead_distance = target
                if ego_ahead_distance < self._release_waypoint_count:
                    return
                self._mpc_target_vehicle_id = None
                self._set_mode(
                    False,
                    f"ego passed vehicle '{vehicle_id}' by {ego_ahead_distance} waypoints")
                return

        self._mpc_target_vehicle_id = None
        self._set_mode(False, "no vehicle within switch range")

    def _vehicle_waypoint_relations(
        self, ego_x: float, ego_y: float
    ) -> List[Tuple[str, int, int]]:
        if self._trajectory is None:
            return []

        points = self._trajectory.points
        waypoint_count = len(points)
        if waypoint_count == 0:
            return []

        ego_idx = nearest_waypoint_index(points, ego_x, ego_y)
        if ego_idx is None:
            return []

        relations: List[Tuple[str, int, int]] = []
        for vehicle_id, x, y in self._vehicles:
            vehicle_idx = nearest_waypoint_index(points, x, y)
            if vehicle_idx is None:
                continue
            vehicle_ahead_distance = forward_waypoint_distance(
                ego_idx, vehicle_idx, waypoint_count)
            ego_ahead_distance = forward_waypoint_distance(
                vehicle_idx, ego_idx, waypoint_count)
            relations.append((vehicle_id, vehicle_ahead_distance, ego_ahead_distance))
        return relations

    def _set_mode(self, use_mpc: bool, reason: str) -> None:
        if use_mpc == self._use_mpc:
            return
        self._use_mpc = use_mpc
        if not use_mpc:
            self._mpc_target_vehicle_id = None
        mode = "MPC" if use_mpc else "Pure Pursuit"
        self.get_logger().info(f"Switched to {mode}: {reason}")

        cmd = self._last_mpc_cmd if use_mpc else self._last_pp_cmd
        if cmd is not None:
            self._pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlCmdMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
