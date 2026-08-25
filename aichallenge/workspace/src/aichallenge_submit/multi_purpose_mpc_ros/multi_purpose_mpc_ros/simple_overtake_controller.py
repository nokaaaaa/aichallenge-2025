#!/usr/bin/env python3

import math
import os
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_planning_msgs.msg import Trajectory
from nav_msgs.msg import Odometry
from v2x_msgs.msg import V2XVehiclePositionArray

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    forward_waypoint_distance,
    nearest_waypoint_index,
)


class SimpleOvertakeController(Node):
    def __init__(self) -> None:
        super().__init__("simple_overtake_controller")

        self.declare_parameter("output_cmd_topic", "/control/command/mpc_cmd")
        self.declare_parameter("kinematics_topic", "/localization/kinematic_state")
        self.declare_parameter("trajectory_topic", "/planning/scenario_planning/trajectory")
        self.declare_parameter("v2x_topic", "/v2x/vehicle_positions")
        self.declare_parameter("ego_vehicle_id", os.environ.get("VEHICLE_ID", ""))
        self.declare_parameter("wheel_base", 1.087)
        self.declare_parameter("lookahead_waypoints", 6)
        self.declare_parameter("vehicle_search_waypoints", 12)
        self.declare_parameter("overtake_offset", 2.0)
        self.declare_parameter("default_overtake_side", "right")
        self.declare_parameter("target_speed", 8.333333333333334)
        self.declare_parameter("kp_accel", 1.2)
        self.declare_parameter("min_acceleration", -2.0)
        self.declare_parameter("max_acceleration", 3.0)
        self.declare_parameter("steering_limit", 0.64)

        self._ego_vehicle_id = str(self.get_parameter("ego_vehicle_id").value)
        self._wheel_base = float(self.get_parameter("wheel_base").value)
        self._lookahead_waypoints = int(self.get_parameter("lookahead_waypoints").value)
        self._vehicle_search_waypoints = int(self.get_parameter("vehicle_search_waypoints").value)
        self._overtake_offset = float(self.get_parameter("overtake_offset").value)
        self._default_side = -1.0 if str(
            self.get_parameter("default_overtake_side").value).lower() == "right" else 1.0
        self._target_speed = float(self.get_parameter("target_speed").value)
        self._kp_accel = float(self.get_parameter("kp_accel").value)
        self._min_accel = float(self.get_parameter("min_acceleration").value)
        self._max_accel = float(self.get_parameter("max_acceleration").value)
        self._steering_limit = float(self.get_parameter("steering_limit").value)

        self._odom: Optional[Odometry] = None
        self._trajectory: Optional[Trajectory] = None
        self._vehicles: List[Tuple[str, float, float]] = []

        output_topic = str(self.get_parameter("output_cmd_topic").value)
        kinematics_topic = str(self.get_parameter("kinematics_topic").value)
        trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        v2x_topic = str(self.get_parameter("v2x_topic").value)

        self._pub = self.create_publisher(AckermannControlCommand, output_topic, 1)
        self.create_subscription(Odometry, kinematics_topic, self._odom_callback, 1)
        trajectory_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Trajectory, trajectory_topic, self._trajectory_callback, trajectory_qos)
        self.create_subscription(V2XVehiclePositionArray, v2x_topic, self._v2x_callback, 1)
        self.create_timer(0.05, self._on_timer)

        self.get_logger().info(
            f"Simple overtake controller publishing '{output_topic}' with "
            f"offset={self._overtake_offset:.1f}m")

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg

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

    def _on_timer(self) -> None:
        if self._odom is None or self._trajectory is None or not self._trajectory.points:
            return

        points = self._trajectory.points
        ego_x = float(self._odom.pose.pose.position.x)
        ego_y = float(self._odom.pose.pose.position.y)
        ego_idx = nearest_waypoint_index(points, ego_x, ego_y)
        if ego_idx is None:
            return

        target_vehicle = self._nearest_vehicle_ahead(ego_idx)
        offset_side = self._default_side
        if target_vehicle is not None:
            offset_side = self._overtake_side_for_vehicle(target_vehicle)

        lookahead_idx = (ego_idx + self._lookahead_waypoints) % len(points)
        base_x, base_y, path_yaw = self._path_state(lookahead_idx)
        left_x = -math.sin(path_yaw)
        left_y = math.cos(path_yaw)
        target_x = base_x + offset_side * self._overtake_offset * left_x
        target_y = base_y + offset_side * self._overtake_offset * left_y

        yaw = self._odom_yaw()
        dx = target_x - ego_x
        dy = target_y - ego_y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        lookahead_distance = max(math.hypot(local_x, local_y), 1e-3)
        alpha = math.atan2(local_y, local_x)
        steering = math.atan2(2.0 * self._wheel_base * math.sin(alpha), lookahead_distance)
        steering = max(-self._steering_limit, min(self._steering_limit, steering))

        current_speed = float(self._odom.twist.twist.linear.x)
        path_speed = float(points[lookahead_idx].longitudinal_velocity_mps)
        speed = max(path_speed, self._target_speed)
        accel = self._kp_accel * (speed - current_speed)
        accel = max(self._min_accel, min(self._max_accel, accel))

        self._publish(speed, accel, steering)

    def _nearest_vehicle_ahead(self, ego_idx: int) -> Optional[Tuple[str, float, float]]:
        if self._trajectory is None:
            return None
        points = self._trajectory.points
        candidates = []
        for vehicle_id, x, y in self._vehicles:
            vehicle_idx = nearest_waypoint_index(points, x, y)
            if vehicle_idx is None:
                continue
            ahead = forward_waypoint_distance(ego_idx, vehicle_idx, len(points))
            if ahead <= self._vehicle_search_waypoints:
                candidates.append((ahead, vehicle_id, x, y))
        if not candidates:
            return None
        _ahead, vehicle_id, x, y = min(candidates, key=lambda item: item[0])
        return vehicle_id, x, y

    def _overtake_side_for_vehicle(self, vehicle: Tuple[str, float, float]) -> float:
        if self._trajectory is None:
            return self._default_side
        _vehicle_id, x, y = vehicle
        points = self._trajectory.points
        vehicle_idx = nearest_waypoint_index(points, x, y)
        if vehicle_idx is None:
            return self._default_side
        path_x, path_y, path_yaw = self._path_state(vehicle_idx)
        left_x = -math.sin(path_yaw)
        left_y = math.cos(path_yaw)
        lateral = (x - path_x) * left_x + (y - path_y) * left_y
        if abs(lateral) < 0.2:
            return self._default_side
        return -1.0 if lateral > 0.0 else 1.0

    def _path_state(self, idx: int) -> Tuple[float, float, float]:
        points = self._trajectory.points
        point = points[idx]
        next_point = points[(idx + 1) % len(points)]
        x = float(point.pose.position.x)
        y = float(point.pose.position.y)
        yaw = math.atan2(
            float(next_point.pose.position.y) - y,
            float(next_point.pose.position.x) - x,
        )
        return x, y, yaw

    def _odom_yaw(self) -> float:
        q = self._odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _publish(self, speed: float, acceleration: float, steering: float) -> None:
        cmd = AckermannControlCommand()
        stamp = self.get_clock().now().to_msg()
        cmd.stamp = stamp
        cmd.longitudinal.stamp = stamp
        cmd.longitudinal.speed = speed
        cmd.longitudinal.acceleration = acceleration
        cmd.lateral.stamp = stamp
        cmd.lateral.steering_tire_angle = steering
        self._pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimpleOvertakeController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
