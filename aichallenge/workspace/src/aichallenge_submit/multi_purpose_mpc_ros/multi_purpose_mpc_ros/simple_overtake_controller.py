#!/usr/bin/env python3

import math
import os
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_planning_msgs.msg import Trajectory
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from v2x_msgs.msg import V2XVehiclePositionArray

from multi_purpose_mpc_ros.common import resolve_package_path
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    forward_waypoint_distance,
    nearest_waypoint_index,
)


class GeometricOvertakePlanner:
    def __init__(
        self,
        map_yaml_path: str,
    ) -> None:
        self._map = Map(map_yaml_path)

    def plan(
        self,
        reference_xy: List[Tuple[float, float]],
        target_vehicle_xy: Tuple[float, float],
        target_waypoint_index: int,
    ) -> Optional[List[Tuple[float, float]]]:
        waypoint_count = len(reference_xy)
        if waypoint_count < 42:
            return None
        j_index = (target_waypoint_index - 30) % waypoint_count
        l_index = (target_waypoint_index + 10) % waypoint_count
        k = self._compute_k(target_vehicle_xy, reference_xy[target_waypoint_index])
        if k is None:
            return None

        # J and L are retained as anchors; the original inclusive J..L section
        # is replaced by equally spaced samples of J-K and K-L.
        j = reference_xy[j_index]
        l = reference_xy[l_index]
        spacing = self._nominal_spacing(reference_xy)
        segment_jk = self._interpolate(j, k, max(1, int(round(self._distance(j, k) / spacing))))
        segment_kl = self._interpolate(k, l, max(1, int(round(self._distance(k, l) / spacing))))
        replaced = segment_jk[:-1] + segment_kl
        path = []
        for offset in range(waypoint_count):
            idx = (j_index + offset) % waypoint_count
            if idx == j_index:
                path.extend(replaced)
            if offset <= 40:
                continue
            path.append(reference_xy[idx])
        # Keep the cyclic route ordered from J through L and back to J.
        return path

    def _compute_k(
        self, vehicle_xy: Tuple[float, float], waypoint_xy: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        dx = waypoint_xy[0] - vehicle_xy[0]
        dy = waypoint_xy[1] - vehicle_xy[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return None
        direction = (dx / length, dy / length)
        intersections = []
        for sign in (-1.0, 1.0):
            distance = self._distance_to_wall(vehicle_xy, (sign * direction[0], sign * direction[1]))
            if distance is not None:
                intersections.append(sign * distance)
        if len(intersections) != 2:
            return None
        far_distance = max(intersections, key=abs)
        return (
            vehicle_xy[0] + direction[0] * far_distance * 0.5,
            vehicle_xy[1] + direction[1] * far_distance * 0.5,
        )

    def _distance_to_wall(
        self, origin: Tuple[float, float], direction: Tuple[float, float]
    ) -> Optional[float]:
        step = max(self._map.resolution * 0.5, 0.01)
        distance = 0.0
        while distance < 200.0:
            distance += step
            x = origin[0] + direction[0] * distance
            y = origin[1] + direction[1] * distance
            cell = self._world_to_map(x, y)
            if cell is None:
                return distance
            mx, my = cell
            if self._map.data[my, mx] == 0:
                return distance
        return None

    def _world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        mx = int((x - self._map.origin[0]) / self._map.resolution + 0.5)
        my = int((self._map.height - 1) - (y - self._map.origin[1]) / self._map.resolution + 0.5)
        if 0 <= mx < self._map.width and 0 <= my < self._map.height:
            return mx, my
        return None

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    @classmethod
    def _nominal_spacing(cls, reference_xy: List[Tuple[float, float]]) -> float:
        distances = [
            cls._distance(reference_xy[i], reference_xy[(i + 1) % len(reference_xy)])
            for i in range(len(reference_xy))
        ]
        usable = [distance for distance in distances if distance > 1e-3]
        return max(sum(usable) / max(len(usable), 1), 0.05)

    @staticmethod
    def _interpolate(
        start: Tuple[float, float], end: Tuple[float, float], count: int
    ) -> List[Tuple[float, float]]:
        return [
            (
                start[0] + (end[0] - start[0]) * i / count,
                start[1] + (end[1] - start[1]) * i / count,
            )
            for i in range(count + 1)
        ]


class SimpleOvertakeController(Node):
    def __init__(self) -> None:
        super().__init__("simple_overtake_controller")

        self.declare_parameter("output_cmd_topic", "/control/command/mpc_cmd")
        self.declare_parameter("kinematics_topic", "/localization/kinematic_state")
        self.declare_parameter("trajectory_topic", "/planning/scenario_planning/trajectory")
        self.declare_parameter("v2x_topic", "/v2x/vehicle_positions")
        self.declare_parameter("ego_vehicle_id", os.environ.get("VEHICLE_ID", ""))
        self.declare_parameter("enabled_ego_vehicle_id", "P1")
        self.declare_parameter("map_yaml_path", "env/final_ver3/occupancy_grid_map.yaml")
        self.declare_parameter("wheel_base", 1.087)
        self.declare_parameter("lookahead_waypoints", 6)
        self.declare_parameter("lookahead_distance", 3.0)
        self.declare_parameter("vehicle_search_waypoints", 12)
        self.declare_parameter("target_speed", 8.333333333333334)
        self.declare_parameter("kp_accel", 1.2)
        self.declare_parameter("min_acceleration", -2.0)
        self.declare_parameter("max_acceleration", 3.0)
        self.declare_parameter("steering_limit", 0.64)
        self.declare_parameter("debug_path_topic", "/planning/overtake/target_path")
        self.declare_parameter("reference_path_topic", "/planning/overtake/reference_path")

        self._ego_vehicle_id = str(self.get_parameter("ego_vehicle_id").value)
        self._enabled_ego_vehicle_id = str(self.get_parameter("enabled_ego_vehicle_id").value)
        self._wheel_base = float(self.get_parameter("wheel_base").value)
        self._lookahead_waypoints = int(self.get_parameter("lookahead_waypoints").value)
        self._lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self._vehicle_search_waypoints = int(self.get_parameter("vehicle_search_waypoints").value)
        self._target_speed = float(self.get_parameter("target_speed").value)
        self._kp_accel = float(self.get_parameter("kp_accel").value)
        self._min_accel = float(self.get_parameter("min_acceleration").value)
        self._max_accel = float(self.get_parameter("max_acceleration").value)
        self._steering_limit = float(self.get_parameter("steering_limit").value)
        self._planned_path: List[Tuple[float, float]] = []
        self._planned_for_vehicle_id: Optional[str] = None
        self._last_plan_time = None

        self._odom: Optional[Odometry] = None
        self._trajectory: Optional[Trajectory] = None
        self._vehicles: List[Tuple[str, float, float]] = []

        output_topic = str(self.get_parameter("output_cmd_topic").value)
        debug_path_topic = str(self.get_parameter("debug_path_topic").value)
        reference_path_topic = str(self.get_parameter("reference_path_topic").value)
        kinematics_topic = str(self.get_parameter("kinematics_topic").value)
        trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        v2x_topic = str(self.get_parameter("v2x_topic").value)
        map_yaml_path = resolve_package_path(
            str(self.get_parameter("map_yaml_path").value),
            get_package_share_directory("multi_purpose_mpc_ros"),
        )
        self._planner = GeometricOvertakePlanner(map_yaml_path=map_yaml_path)

        self._pub = self.create_publisher(AckermannControlCommand, output_topic, 1)
        path_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._path_pub = self.create_publisher(Path, debug_path_topic, path_qos)
        self._reference_path_pub = self.create_publisher(
            Path, reference_path_topic, path_qos
        )
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
            f"Costmap overtake controller publishing '{output_topic}' for "
            f"ego='{self._enabled_ego_vehicle_id}' using map '{map_yaml_path}', "
            f"debug path='{debug_path_topic}', reference path='{reference_path_topic}'")

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
        if self._enabled_ego_vehicle_id and self._ego_vehicle_id != self._enabled_ego_vehicle_id:
            return

        points = self._trajectory.points
        reference_xy = [
            (float(point.pose.position.x), float(point.pose.position.y))
            for point in points
        ]
        self._publish_reference_path(reference_xy)
        ego_x = float(self._odom.pose.pose.position.x)
        ego_y = float(self._odom.pose.pose.position.y)
        ego_idx = nearest_waypoint_index(points, ego_x, ego_y)
        if ego_idx is None:
            return

        target_vehicle = self._nearest_vehicle_ahead(ego_idx)
        if target_vehicle is None:
            self._planned_path = []
            self._planned_for_vehicle_id = None
            self._publish_path([])
            return

        vehicle_id, target_x, target_y = target_vehicle
        if self._needs_replan(vehicle_id):
            self._planned_path = self._plan_overtake_path((target_x, target_y))
            self._planned_for_vehicle_id = vehicle_id
            self._last_plan_time = self.get_clock().now()
            self._publish_path(self._planned_path)
            self.get_logger().info(
                f"Published overtake path with {len(self._planned_path)} points "
                f"for vehicle '{vehicle_id}'"
            )
        if not self._planned_path:
            return

        target_x, target_y = self._lookahead_path_point(ego_x, ego_y)

        yaw = self._odom_yaw()
        dx = target_x - ego_x
        dy = target_y - ego_y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if local_x <= 0.0:
            self.get_logger().warn("Overtake target point is behind ego; skipping command")
            return
        lookahead_distance = max(math.hypot(local_x, local_y), 1e-3)
        alpha = math.atan2(local_y, local_x)
        steering = math.atan2(2.0 * self._wheel_base * math.sin(alpha), lookahead_distance)
        steering = max(-self._steering_limit, min(self._steering_limit, steering))

        current_speed = float(self._odom.twist.twist.linear.x)
        speed_idx = (ego_idx + self._lookahead_waypoints) % len(points)
        path_speed = float(points[speed_idx].longitudinal_velocity_mps)
        speed = max(path_speed, self._target_speed)
        accel = self._kp_accel * (speed - current_speed)
        accel = max(self._min_accel, min(self._max_accel, accel))

        self._publish(speed, accel, steering)

    def _plan_overtake_path(
        self, target_vehicle_xy: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        points = self._trajectory.points
        reference_xy = [
            (
                float(point.pose.position.x),
                float(point.pose.position.y),
            )
            for point in points
        ]
        target_waypoint_index = nearest_waypoint_index(
            points, target_vehicle_xy[0], target_vehicle_xy[1]
        )
        if target_waypoint_index is None:
            return []
        planned = self._planner.plan(
            reference_xy,
            target_vehicle_xy,
            target_waypoint_index,
        )
        if planned is None:
            self.get_logger().warn("Geometric overtake planner failed to find a path")
            return []
        return planned

    def _needs_replan(self, vehicle_id: str) -> bool:
        if not self._planned_path or self._planned_for_vehicle_id != vehicle_id:
            return True
        if self._last_plan_time is None:
            return True
        elapsed = (self.get_clock().now() - self._last_plan_time).nanoseconds * 1e-9
        return elapsed >= 0.25

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

    def _lookahead_path_point(self, ego_x: float, ego_y: float) -> Tuple[float, float]:
        nearest_idx = min(
            range(len(self._planned_path)),
            key=lambda i: (self._planned_path[i][0] - ego_x) ** 2
            + (self._planned_path[i][1] - ego_y) ** 2,
        )
        if len(self._planned_path) <= 1:
            return self._planned_path[nearest_idx]

        distance_left = max(self._lookahead_distance, 0.1)
        prev_x, prev_y = self._planned_path[nearest_idx]
        for idx in range(nearest_idx + 1, len(self._planned_path)):
            next_x, next_y = self._planned_path[idx]
            segment_length = math.hypot(next_x - prev_x, next_y - prev_y)
            if segment_length >= distance_left:
                ratio = distance_left / max(segment_length, 1e-6)
                return (
                    prev_x + ratio * (next_x - prev_x),
                    prev_y + ratio * (next_y - prev_y),
                )
            distance_left -= segment_length
            prev_x, prev_y = next_x, next_y

        return self._planned_path[-1]

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

    def _publish_path(self, path_xy: List[Tuple[float, float]]) -> None:
        self._publish_path_message(self._path_pub, path_xy)

    def _publish_reference_path(self, path_xy: List[Tuple[float, float]]) -> None:
        self._publish_path_message(self._reference_path_pub, path_xy)

    def _publish_path_message(self, publisher, path_xy: List[Tuple[float, float]]) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        if self._trajectory is not None and self._trajectory.header.frame_id:
            msg.header.frame_id = self._trajectory.header.frame_id
        else:
            msg.header.frame_id = "map"
        for x, y in path_xy:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        publisher.publish(msg)


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
