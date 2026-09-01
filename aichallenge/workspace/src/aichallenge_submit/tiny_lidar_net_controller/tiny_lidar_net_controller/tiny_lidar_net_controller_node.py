#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import numpy as np

import rclpy
from autoware_auto_control_msgs.msg import AckermannControlCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - import guard for runtime environments
    raise RuntimeError(
        "stable-baselines3 is required to run the PPO controller. "
        "Install it in the ROS runtime environment or run this node in the kart_rl uv environment."
    ) from exc


class TinyLidarNetNode(Node):
    """PPO controller that turns point cloud input into Ackermann commands."""

    def __init__(self):
        super().__init__("tiny_lidar_net_node")

        self.declare_parameter("log_interval_sec", 5.0)
        self.declare_parameter("use_sim_time", True)
        self.declare_parameter("model.path", "")
        self.declare_parameter("model.device", "cpu")
        self.declare_parameter("lidar.angle_min", -1.5666074752807617)
        self.declare_parameter("lidar.angle_max", 1.5707963705062866)
        self.declare_parameter("lidar.range_min", 0.0)
        self.declare_parameter("lidar.range_max", 25.0)
        self.declare_parameter("scan.topic", "/sensing/lidar/scan")
        self.declare_parameter("odometry.topic", "/localization/kinematic_state")
        self.declare_parameter("control.topic", "/control/command/control_cmd")
        self.declare_parameter("control.dt", 0.05)
        self.declare_parameter("vehicle.min_speed_mps", 0.0)
        self.declare_parameter("vehicle.max_speed_mps", 4.165)
        self.declare_parameter("vehicle.max_accel_mps2", 3.2)
        self.declare_parameter("vehicle.max_brake_mps2", 5.0)
        self.declare_parameter("vehicle.max_steer_rad", 0.75)
        self.declare_parameter("debug", False)

        self.debug = bool(self.get_parameter("debug").value)
        self.log_interval = float(self.get_parameter("log_interval_sec").value)
        self.control_dt = float(self.get_parameter("control.dt").value)
        self.min_speed = float(self.get_parameter("vehicle.min_speed_mps").value)
        self.max_speed = float(self.get_parameter("vehicle.max_speed_mps").value)
        self.max_accel = float(self.get_parameter("vehicle.max_accel_mps2").value)
        self.max_brake = float(self.get_parameter("vehicle.max_brake_mps2").value)
        self.max_steer = float(self.get_parameter("vehicle.max_steer_rad").value)

        self.angle_min = float(self.get_parameter("lidar.angle_min").value)
        self.angle_max = float(self.get_parameter("lidar.angle_max").value)
        self.range_min = float(self.get_parameter("lidar.range_min").value)
        self.range_max = float(self.get_parameter("lidar.range_max").value)
        self.scan_topic = str(self.get_parameter("scan.topic").value)
        self.odometry_topic = str(self.get_parameter("odometry.topic").value)
        self.control_topic = str(self.get_parameter("control.topic").value)

        model_path = str(self.get_parameter("model.path").value)
        model_device = str(self.get_parameter("model.device").value)
        if not model_path:
            raise ValueError("model.path is empty. Set a PPO .zip model path before starting the node.")

        self.model = PPO.load(model_path, device=model_device)
        obs_shape = getattr(self.model.observation_space, "shape", None)
        if not obs_shape or len(obs_shape) != 1:
            raise ValueError(f"Unsupported PPO observation space: {self.model.observation_space}")
        self.input_dim = int(obs_shape[0])
        self.angle_increment = (self.angle_max - self.angle_min) / max(self.input_dim - 1, 1)

        self.current_speed = 0.0
        self.inference_times: list[float] = []
        self.last_log_time = self.get_clock().now()

        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_scan)
        self.sub_odometry = self.create_subscription(Odometry, self.odometry_topic, self.odometry_callback, qos_reliable)
        self.pub_control = self.create_publisher(AckermannControlCommand, self.control_topic, 1)

        self.get_logger().info(
            f"PPO lidar controller ready: scan={self.scan_topic}, odom={self.odometry_topic}, "
            f"control={self.control_topic}, input_dim={self.input_dim}, model={model_path}"
        )

    def odometry_callback(self, msg: Odometry) -> None:
        self.current_speed = float(msg.twist.twist.linear.x)

    def scan_callback(self, msg: LaserScan) -> None:
        start_time = time.monotonic()

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.range_max, posinf=self.range_max, neginf=self.range_max)
        ranges = np.clip(ranges, self.range_min, self.range_max)
        if len(ranges) != self.input_dim:
            idx = np.linspace(0, max(len(ranges) - 1, 0), self.input_dim, dtype=int)
            ranges = ranges[idx] if len(ranges) else np.full(self.input_dim, self.range_max, dtype=np.float32)
        obs = ranges / self.range_max
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 2:
            self.get_logger().warn("PPO model returned fewer than 2 actions; skipping command publication.")
            return

        target_speed = self.min_speed + 0.5 * (float(action[0]) + 1.0) * (self.max_speed - self.min_speed)
        speed_error = target_speed - self.current_speed
        accel_limit = self.max_accel if speed_error >= 0.0 else self.max_brake
        accel = float(np.clip(speed_error / max(self.control_dt, 1e-3), -accel_limit, accel_limit))
        steer = float(np.clip(float(action[1]) * self.max_steer, -self.max_steer, self.max_steer))

        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = accel
        cmd.lateral.steering_tire_angle = steer
        self.pub_control.publish(cmd)

        if self.debug:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            self.inference_times.append(duration_ms)
            self._log_performance_metrics()

    def _log_performance_metrics(self) -> None:
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_log_time).nanoseconds / 1e9

        if elapsed_sec > self.log_interval:
            if self.inference_times:
                avg_time = float(np.mean(self.inference_times))
                max_time = float(np.max(self.inference_times))
                hz = 1000.0 / avg_time if avg_time > 0.0 else 0.0
                self.get_logger().info(
                    f"DEBUG: Avg inference {avg_time:.2f}ms ({hz:.2f}Hz) | Max {max_time:.2f}ms"
                )
                self.inference_times.clear()
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = TinyLidarNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
