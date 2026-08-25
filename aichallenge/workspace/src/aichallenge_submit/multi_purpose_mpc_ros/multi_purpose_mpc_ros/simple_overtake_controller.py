#!/usr/bin/env python3

import math
import os
import heapq
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_planning_msgs.msg import Trajectory
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from skimage.morphology import binary_dilation, disk
from v2x_msgs.msg import V2XVehiclePositionArray

from multi_purpose_mpc_ros.common import resolve_package_path
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    forward_waypoint_distance,
    nearest_waypoint_index,
)


class CostmapPathPlanner:
    def __init__(
        self,
        map_yaml_path: str,
        vehicle_radius: float,
        ego_radius: float,
        planning_margin: float,
        path_follow_cost: float,
    ) -> None:
        self._map = Map(map_yaml_path)
        self._vehicle_radius = vehicle_radius
        self._ego_radius = ego_radius
        self._planning_margin = planning_margin
        self._path_follow_cost = path_follow_cost
        self._static_inflated_grid = self._build_inflated_static_grid()

    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        reference_xy: List[Tuple[float, float]],
        vehicle_xy: List[Tuple[float, float]],
    ) -> Optional[List[Tuple[float, float]]]:
        grid = self._static_inflated_grid.copy()
        self._paint_vehicle_walls(grid, vehicle_xy)

        sx, sy = self._map.w2m(start_xy[0], start_xy[1])
        gx, gy = self._map.w2m(goal_xy[0], goal_xy[1])
        bounds = self._local_bounds((sx, sy), (gx, gy), reference_xy, vehicle_xy)

        if not self._in_bounds(sx, sy, bounds):
            return None
        if grid[sy, sx] == 0:
            sx, sy = self._nearest_free_cell(grid, sx, sy, bounds) or (sx, sy)
        if grid[gy, gx] == 0:
            nearest_goal = self._nearest_free_cell(grid, gx, gy, bounds)
            if nearest_goal is None:
                return None
            gx, gy = nearest_goal

        path_cells = self._astar(grid, (sx, sy), (gx, gy), bounds, reference_xy)
        if not path_cells:
            return None
        return [self._map.m2w(x, y) for x, y in self._downsample_cells(path_cells)]

    def _build_inflated_static_grid(self) -> np.ndarray:
        radius_px = max(1, int(math.ceil(self._ego_radius / self._map.resolution)))
        occupied = self._map.data == 0
        inflated = binary_dilation(occupied, disk(radius_px))
        return np.where(inflated, 0, self._map.data).astype(np.int8)

    def _paint_vehicle_walls(self, grid: np.ndarray, vehicles: List[Tuple[float, float]]) -> None:
        radius_px = max(1, int(math.ceil(self._vehicle_radius / self._map.resolution)))
        for x, y in vehicles:
            cx, cy = self._map.w2m(x, y)
            self._paint_disc(grid, cx, cy, radius_px)

    @staticmethod
    def _paint_disc(grid: np.ndarray, cx: int, cy: int, radius_px: int) -> None:
        y_min = max(0, cy - radius_px)
        y_max = min(grid.shape[0], cy + radius_px + 1)
        x_min = max(0, cx - radius_px)
        x_max = min(grid.shape[1], cx + radius_px + 1)
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2
        grid[y_min:y_max, x_min:x_max][mask] = 0

    def _local_bounds(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        reference_xy: List[Tuple[float, float]],
        vehicle_xy: List[Tuple[float, float]],
    ) -> Tuple[int, int, int, int]:
        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        for x, y in reference_xy + vehicle_xy:
            mx, my = self._map.w2m(x, y)
            xs.append(mx)
            ys.append(my)
        margin_px = max(4, int(math.ceil(self._planning_margin / self._map.resolution)))
        return (
            max(0, min(xs) - margin_px),
            min(self._map.width - 1, max(xs) + margin_px),
            max(0, min(ys) - margin_px),
            min(self._map.height - 1, max(ys) + margin_px),
        )

    @staticmethod
    def _in_bounds(x: int, y: int, bounds: Tuple[int, int, int, int]) -> bool:
        min_x, max_x, min_y, max_y = bounds
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _nearest_free_cell(
        self,
        grid: np.ndarray,
        x: int,
        y: int,
        bounds: Tuple[int, int, int, int],
    ) -> Optional[Tuple[int, int]]:
        for radius in range(1, 12):
            for yy in range(y - radius, y + radius + 1):
                for xx in range(x - radius, x + radius + 1):
                    if not self._in_bounds(xx, yy, bounds):
                        continue
                    if grid[yy, xx] != 0:
                        return xx, yy
        return None

    def _astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        bounds: Tuple[int, int, int, int],
        reference_xy: List[Tuple[float, float]],
    ) -> Optional[List[Tuple[int, int]]]:
        reference_cells = [self._map.w2m(x, y) for x, y in reference_xy]
        open_heap = [(self._heuristic(start, goal), 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        neighbors = (
            (-1, -1, math.sqrt(2.0)), (0, -1, 1.0), (1, -1, math.sqrt(2.0)),
            (-1, 0, 1.0), (1, 0, 1.0),
            (-1, 1, math.sqrt(2.0)), (0, 1, 1.0), (1, 1, math.sqrt(2.0)),
        )

        while open_heap:
            _f, current_g, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)
            if current_g > g_score.get(current, float("inf")):
                continue

            for dx, dy, step_cost in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                if not self._in_bounds(nx, ny, bounds) or grid[ny, nx] == 0:
                    continue
                tentative = current_g + step_cost + self._reference_cost(nx, ny, reference_cells)
                neighbor = (nx, ny)
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heapq.heappush(
                    open_heap,
                    (tentative + self._heuristic(neighbor, goal), tentative, neighbor),
                )
        return None

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _reference_cost(self, x: int, y: int, reference_cells: List[Tuple[int, int]]) -> float:
        if not reference_cells or self._path_follow_cost <= 0.0:
            return 0.0
        min_dist_sq = min((x - rx) ** 2 + (y - ry) ** 2 for rx, ry in reference_cells)
        return self._path_follow_cost * min_dist_sq

    @staticmethod
    def _reconstruct_path(came_from, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _downsample_cells(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        prev_dx = cells[1][0] - cells[0][0]
        prev_dy = cells[1][1] - cells[0][1]
        for i in range(1, len(cells) - 1):
            dx = cells[i + 1][0] - cells[i][0]
            dy = cells[i + 1][1] - cells[i][1]
            if dx != prev_dx or dy != prev_dy:
                out.append(cells[i])
                prev_dx = dx
                prev_dy = dy
        out.append(cells[-1])
        return out


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
        self.declare_parameter("vehicle_search_waypoints", 12)
        self.declare_parameter("planning_goal_waypoints", 70)
        self.declare_parameter("planning_margin", 4.0)
        self.declare_parameter("vehicle_wall_radius", 1.25)
        self.declare_parameter("ego_collision_radius", 0.85)
        self.declare_parameter("path_follow_cost", 0.001)
        self.declare_parameter("target_speed", 8.333333333333334)
        self.declare_parameter("kp_accel", 1.2)
        self.declare_parameter("min_acceleration", -2.0)
        self.declare_parameter("max_acceleration", 3.0)
        self.declare_parameter("steering_limit", 0.64)

        self._ego_vehicle_id = str(self.get_parameter("ego_vehicle_id").value)
        self._enabled_ego_vehicle_id = str(self.get_parameter("enabled_ego_vehicle_id").value)
        self._wheel_base = float(self.get_parameter("wheel_base").value)
        self._lookahead_waypoints = int(self.get_parameter("lookahead_waypoints").value)
        self._vehicle_search_waypoints = int(self.get_parameter("vehicle_search_waypoints").value)
        self._planning_goal_waypoints = int(self.get_parameter("planning_goal_waypoints").value)
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
        kinematics_topic = str(self.get_parameter("kinematics_topic").value)
        trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        v2x_topic = str(self.get_parameter("v2x_topic").value)
        map_yaml_path = resolve_package_path(
            str(self.get_parameter("map_yaml_path").value),
            get_package_share_directory("multi_purpose_mpc_ros"),
        )
        self._planner = CostmapPathPlanner(
            map_yaml_path=map_yaml_path,
            vehicle_radius=float(self.get_parameter("vehicle_wall_radius").value),
            ego_radius=float(self.get_parameter("ego_collision_radius").value),
            planning_margin=float(self.get_parameter("planning_margin").value),
            path_follow_cost=float(self.get_parameter("path_follow_cost").value),
        )

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
            f"Costmap overtake controller publishing '{output_topic}' for "
            f"ego='{self._enabled_ego_vehicle_id}' using map '{map_yaml_path}'")

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
        ego_x = float(self._odom.pose.pose.position.x)
        ego_y = float(self._odom.pose.pose.position.y)
        ego_idx = nearest_waypoint_index(points, ego_x, ego_y)
        if ego_idx is None:
            return

        target_vehicle = self._nearest_vehicle_ahead(ego_idx)
        if target_vehicle is None:
            self._planned_path = []
            self._planned_for_vehicle_id = None
            return

        vehicle_id, _target_x, _target_y = target_vehicle
        if self._needs_replan(vehicle_id):
            self._planned_path = self._plan_overtake_path(ego_idx)
            self._planned_for_vehicle_id = vehicle_id
            self._last_plan_time = self.get_clock().now()
        if not self._planned_path:
            return

        target_x, target_y = self._lookahead_path_point(ego_x, ego_y)

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
        speed_idx = (ego_idx + self._lookahead_waypoints) % len(points)
        path_speed = float(points[speed_idx].longitudinal_velocity_mps)
        speed = max(path_speed, self._target_speed)
        accel = self._kp_accel * (speed - current_speed)
        accel = max(self._min_accel, min(self._max_accel, accel))

        self._publish(speed, accel, steering)

    def _plan_overtake_path(self, ego_idx: int) -> List[Tuple[float, float]]:
        points = self._trajectory.points
        goal_idx = (ego_idx + self._planning_goal_waypoints) % len(points)
        ego_x = float(self._odom.pose.pose.position.x)
        ego_y = float(self._odom.pose.pose.position.y)
        goal_x = float(points[goal_idx].pose.position.x)
        goal_y = float(points[goal_idx].pose.position.y)
        reference_xy = [
            (
                float(points[(ego_idx + i) % len(points)].pose.position.x),
                float(points[(ego_idx + i) % len(points)].pose.position.y),
            )
            for i in range(self._planning_goal_waypoints + 1)
        ]
        vehicle_xy = self._planning_vehicle_positions(ego_idx)
        planned = self._planner.plan(
            (ego_x, ego_y),
            (goal_x, goal_y),
            reference_xy,
            vehicle_xy,
        )
        if planned is None:
            self.get_logger().warn("Costmap overtake planner failed to find a path")
            return []
        return planned

    def _needs_replan(self, vehicle_id: str) -> bool:
        if not self._planned_path or self._planned_for_vehicle_id != vehicle_id:
            return True
        if self._last_plan_time is None:
            return True
        elapsed = (self.get_clock().now() - self._last_plan_time).nanoseconds * 1e-9
        return elapsed >= 0.25

    def _planning_vehicle_positions(self, ego_idx: int) -> List[Tuple[float, float]]:
        points = self._trajectory.points
        vehicle_xy: List[Tuple[float, float]] = []
        for _vehicle_id, x, y in self._vehicles:
            vehicle_idx = nearest_waypoint_index(points, x, y)
            if vehicle_idx is None:
                continue
            ahead = forward_waypoint_distance(ego_idx, vehicle_idx, len(points))
            if ahead <= self._planning_goal_waypoints + self._vehicle_search_waypoints:
                vehicle_xy.append((x, y))
        return vehicle_xy

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
        target_idx = min(nearest_idx + self._lookahead_waypoints, len(self._planned_path) - 1)
        return self._planned_path[target_idx]

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
