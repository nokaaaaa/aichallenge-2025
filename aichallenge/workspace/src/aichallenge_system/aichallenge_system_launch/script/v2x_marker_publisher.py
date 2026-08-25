#!/usr/bin/env python3
import math
import os

import rclpy
import rclpy.node
from autoware_auto_planning_msgs.msg import Trajectory
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray
from v2x_msgs.msg import V2XVehiclePositionArray


VEHICLE_COLORS = {
    "d1": (0.2, 0.4, 1.0),
    "d2": (1.0, 0.9, 0.2),
    "d3": (0.2, 1.0, 0.2),
    "d4": (1.0, 0.2, 0.2),
}
DEFAULT_COLOR = (1.0, 1.0, 1.0)
SPHERE_DIAMETER = 1.5
ALPHA = 0.9
LIFETIME_SEC = 1
MODE_PP = "Pure Pursuit"
MODE_OVERTAKE = "Overtake"
MODE_NO_DATA = "No data"
DEFAULT_VEHICLE_IDS = ["d1", "d2", "d3", "d4"]


def nearest_waypoint_index(points, x: float, y: float):
    nearest_idx = None
    nearest_dist_sq = math.inf
    for idx, point in enumerate(points):
        px = float(point.pose.position.x)
        py = float(point.pose.position.y)
        dist_sq = (px - x) ** 2 + (py - y) ** 2
        if dist_sq < nearest_dist_sq:
            nearest_idx = idx
            nearest_dist_sq = dist_sq
    return nearest_idx


def forward_waypoint_distance(from_index: int, to_index: int, waypoint_count: int) -> int:
    return ((to_index % waypoint_count) - (from_index % waypoint_count)) % waypoint_count


def default_ego_vehicle_id() -> str:
    vehicle_id = os.environ.get("VEHICLE_ID", "").strip()
    if vehicle_id:
        return vehicle_id
    domain_id = os.environ.get("ROS_DOMAIN_ID", "").strip()
    if domain_id.isdigit() and int(domain_id) > 0:
        return f"d{int(domain_id)}"
    return ""


class V2XMarkerPublisherNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("v2x_marker_publisher")
        self.declare_parameter("show_mode_labels", True)
        self.declare_parameter("publish_mode_summary", True)
        self.declare_parameter("vehicle_ids", DEFAULT_VEHICLE_IDS)
        self.declare_parameter("ego_vehicle_id", default_ego_vehicle_id())
        self.declare_parameter("kinematics_topic", "/localization/kinematic_state")
        self.declare_parameter("trajectory_topic", "/planning/scenario_planning/trajectory")
        self.declare_parameter("switch_waypoint_count", 6)
        self.declare_parameter("release_waypoint_count", 1)
        self.declare_parameter("label_frame_id", "map")
        self.declare_parameter("label_origin_x", 0.0)
        self.declare_parameter("label_origin_y", 0.0)
        self.declare_parameter("label_origin_z", 3.0)
        self.declare_parameter("label_line_spacing", 1.0)
        self.declare_parameter("label_scale", 0.7)

        self._show_mode_labels = self.get_parameter(
            "show_mode_labels").get_parameter_value().bool_value
        self._publish_mode_summary = self.get_parameter(
            "publish_mode_summary").get_parameter_value().bool_value
        self._vehicle_ids = list(self.get_parameter(
            "vehicle_ids").get_parameter_value().string_array_value)
        if not self._vehicle_ids:
            self._vehicle_ids = DEFAULT_VEHICLE_IDS
        self._ego_vehicle_id = str(self.get_parameter("ego_vehicle_id").value)
        kinematics_topic = str(self.get_parameter("kinematics_topic").value)
        trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        self._switch_waypoint_count = int(self.get_parameter("switch_waypoint_count").value)
        self._release_waypoint_count = int(self.get_parameter("release_waypoint_count").value)
        self._label_frame_id = self.get_parameter(
            "label_frame_id").get_parameter_value().string_value
        self._label_origin_x = self.get_parameter(
            "label_origin_x").get_parameter_value().double_value
        self._label_origin_y = self.get_parameter(
            "label_origin_y").get_parameter_value().double_value
        self._label_origin_z = self.get_parameter(
            "label_origin_z").get_parameter_value().double_value
        self._label_line_spacing = self.get_parameter(
            "label_line_spacing").get_parameter_value().double_value
        self._label_scale = self.get_parameter(
            "label_scale").get_parameter_value().double_value
        self._mode_by_vehicle = {}
        self._mode_target_by_vehicle = {}
        self._active_vehicle_ids = set()
        self._odom = None
        self._trajectory = None

        self.sub = self.create_subscription(
            V2XVehiclePositionArray, "/v2x/vehicle_positions", self.callback, 1)
        self.odom_sub = self.create_subscription(
            Odometry, kinematics_topic, self._odom_callback, 1)
        trajectory_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.trajectory_sub = self.create_subscription(
            Trajectory, trajectory_topic, self._trajectory_callback, trajectory_qos)
        self.pub = self.create_publisher(
            MarkerArray, "/v2x/vehicle_positions/markers", 1)
        self.mode_summary_pub = self.create_publisher(
            String, "/v2x/vehicle_modes/text", 1)
        self.create_timer(0.5, self._publish_mode_summary_timer)

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg
        if self._ego_vehicle_id:
            self._active_vehicle_ids.add(self._ego_vehicle_id)

    def _trajectory_callback(self, msg: Trajectory) -> None:
        self._trajectory = msg

    def callback(self, msg: V2XVehiclePositionArray) -> None:
        markers = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        if self._show_mode_labels or self._publish_mode_summary:
            self._update_modes(msg)

        sorted_vehicles = sorted(msg.vehicles, key=lambda vehicle: vehicle.vehicle_id)

        for index, vehicle in enumerate(sorted_vehicles):
            markers.markers.append(self._build_marker(msg, vehicle, index))

        if self._show_mode_labels:
            for index, vehicle in enumerate(sorted_vehicles):
                markers.markers.append(self._build_mode_label(msg, vehicle, index))

        self.pub.publish(markers)
        if self._publish_mode_summary:
            self._publish_mode_summary_text(sorted_vehicles)

    def _update_modes(self, msg: V2XVehiclePositionArray) -> None:
        positions = self._vehicle_positions(msg)
        self._active_vehicle_ids = set(positions.keys())
        if self._trajectory is None or not self._trajectory.points:
            return

        for vehicle_id, (x, y) in positions.items():
            vehicle_idx = nearest_waypoint_index(self._trajectory.points, x, y)
            if vehicle_idx is None:
                continue

            switch_candidate = None
            current_target = self._mode_target_by_vehicle.get(vehicle_id)
            target_relation = None
            for other_id, (other_x, other_y) in positions.items():
                if other_id == vehicle_id:
                    continue
                other_idx = nearest_waypoint_index(self._trajectory.points, other_x, other_y)
                if other_idx is None:
                    continue
                other_ahead_distance = forward_waypoint_distance(
                    vehicle_idx, other_idx, len(self._trajectory.points))
                vehicle_ahead_distance = forward_waypoint_distance(
                    other_idx, vehicle_idx, len(self._trajectory.points))
                relation = (other_id, other_ahead_distance, vehicle_ahead_distance)
                if other_ahead_distance <= self._switch_waypoint_count:
                    if switch_candidate is None or other_ahead_distance < switch_candidate[1]:
                        switch_candidate = relation
                if other_id == current_target:
                    target_relation = relation

            if switch_candidate is not None:
                other_id, _other_ahead_distance, _vehicle_ahead_distance = switch_candidate
                self._mode_target_by_vehicle[vehicle_id] = other_id
                self._mode_by_vehicle[vehicle_id] = MODE_OVERTAKE
                continue

            if self._mode_by_vehicle.get(vehicle_id, MODE_PP) == MODE_OVERTAKE:
                if target_relation is not None and target_relation[2] < self._release_waypoint_count:
                    continue
                self._mode_target_by_vehicle.pop(vehicle_id, None)

            self._mode_by_vehicle[vehicle_id] = MODE_PP

    def _vehicle_positions(self, msg: V2XVehiclePositionArray) -> dict:
        positions = {
            vehicle.vehicle_id: (vehicle.position.x, vehicle.position.y)
            for vehicle in msg.vehicles
        }
        if self._ego_vehicle_id and self._odom is not None:
            positions[self._ego_vehicle_id] = (
                self._odom.pose.pose.position.x,
                self._odom.pose.pose.position.y,
            )
        return positions

    def _publish_mode_summary_text(self, vehicles) -> None:
        vehicle_ids = self._summary_vehicle_ids(vehicles)
        lines = [
            f"{vehicle_id}: {self._summary_mode(vehicle_id)}"
            for vehicle_id in vehicle_ids
        ]
        msg = String()
        msg.data = "\n".join(lines)
        self.mode_summary_pub.publish(msg)

    def _summary_vehicle_ids(self, vehicles) -> list:
        configured_ids = list(self._vehicle_ids)
        observed_ids = sorted(vehicle.vehicle_id for vehicle in vehicles)
        if self._ego_vehicle_id and self._ego_vehicle_id not in configured_ids:
            configured_ids.insert(0, self._ego_vehicle_id)
        extra_ids = [vehicle_id for vehicle_id in observed_ids if vehicle_id not in configured_ids]
        return configured_ids + extra_ids

    def _publish_mode_summary_timer(self) -> None:
        if not self._publish_mode_summary:
            return
        self._publish_mode_summary_text([])

    def _summary_mode(self, vehicle_id: str) -> str:
        if vehicle_id not in self._active_vehicle_ids:
            return MODE_NO_DATA
        return self._mode_by_vehicle.get(vehicle_id, MODE_PP)

    def _build_marker(self, array_msg, vehicle, index: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = vehicle.header.frame_id or array_msg.header.frame_id or "map"
        marker.header.stamp = vehicle.header.stamp
        marker.ns = "v2x_vehicles"
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = vehicle.position.x
        marker.pose.position.y = vehicle.position.y
        marker.pose.position.z = vehicle.position.z
        marker.pose.orientation.w = 1.0
        marker.scale.x = SPHERE_DIAMETER
        marker.scale.y = SPHERE_DIAMETER
        marker.scale.z = SPHERE_DIAMETER
        r, g, b = VEHICLE_COLORS.get(vehicle.vehicle_id, DEFAULT_COLOR)
        marker.color = ColorRGBA(r=r, g=g, b=b, a=ALPHA)
        marker.lifetime = Duration(sec=LIFETIME_SEC, nanosec=0)
        return marker

    def _build_mode_label(self, array_msg, vehicle, index: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._label_frame_id or array_msg.header.frame_id or "map"
        marker.header.stamp = array_msg.header.stamp
        marker.ns = "v2x_vehicle_modes"
        marker.id = 1000 + index
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = self._label_origin_x
        marker.pose.position.y = self._label_origin_y - index * self._label_line_spacing
        marker.pose.position.z = self._label_origin_z
        marker.pose.orientation.w = 1.0
        marker.scale.z = self._label_scale

        mode = self._mode_by_vehicle.get(vehicle.vehicle_id, MODE_PP)
        marker.text = f"{vehicle.vehicle_id}: {mode}"
        marker.color = self._mode_color(mode)
        marker.lifetime = Duration(sec=LIFETIME_SEC, nanosec=0)
        return marker

    @staticmethod
    def _mode_color(mode: str) -> ColorRGBA:
        if mode == MODE_OVERTAKE:
            return ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        return ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(V2XMarkerPublisherNode())
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
