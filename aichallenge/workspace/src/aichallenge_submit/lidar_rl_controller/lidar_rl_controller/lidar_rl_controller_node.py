#!/usr/bin/env python3
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

import rclpy
from autoware_auto_control_msgs.msg import AckermannControlCommand
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


SIM_ACCEL_LIMIT_MPS2 = 3.0


def resolve_latest_timestamped_policy(base_path: str) -> Path:
    base = Path(base_path).expanduser()
    stem = base.with_suffix("") if base.suffix else base
    pattern = re.compile(rf"^{re.escape(stem.name)}_(\d{{8}}-\d{{6}})(?:_(\d+))?$")
    candidates: list[tuple[str, int, Path]] = []
    for run_dir in stem.parent.iterdir() if stem.parent.exists() else []:
        if not run_dir.is_dir():
            continue
        match = pattern.fullmatch(run_dir.name)
        if not match:
            continue
        policy_path = run_dir / f"{stem.name}_policy.npz"
        if policy_path.exists():
            candidates.append((match.group(1), int(match.group(2) or 0), policy_path))

    if not candidates:
        raise FileNotFoundError(
            f"No timestamped policy found for {stem}. "
            f"Expected {stem.parent}/{stem.name}_YYYYMMDD-HHMMSS/{stem.name}_policy.npz"
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def resolve_policy_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.suffix:
        return path
    return resolve_latest_timestamped_policy(model_path)


class NumpyPpoPolicy:
    """Deterministic SB3 MlpPolicy actor exported by kart_rl.train."""

    def __init__(self, path: str):
        weights = np.load(path)
        self.w0 = weights["mlp_extractor.policy_net.0.weight"].astype(np.float32)
        self.b0 = weights["mlp_extractor.policy_net.0.bias"].astype(np.float32)
        self.w2 = weights["mlp_extractor.policy_net.2.weight"].astype(np.float32)
        self.b2 = weights["mlp_extractor.policy_net.2.bias"].astype(np.float32)
        self.wa = weights["action_net.weight"].astype(np.float32)
        self.ba = weights["action_net.bias"].astype(np.float32)
        self.input_dim = int(self.w0.shape[1])

    def predict(self, obs: np.ndarray) -> np.ndarray:
        x = np.tanh(self.w0 @ obs + self.b0)
        x = np.tanh(self.w2 @ x + self.b2)
        return np.clip(self.wa @ x + self.ba, -1.0, 1.0)


class LidarRlControllerNode(Node):
    """PPO LiDAR controller that publishes Ackermann commands."""

    def __init__(self):
        super().__init__("lidar_rl_controller_node")

        self.declare_parameter("log_interval_sec", 5.0)
        self.declare_parameter("model.path", "")
        self.declare_parameter("lidar.range_min", 0.0)
        self.declare_parameter("lidar.range_max", 25.0)
        self.declare_parameter("scan.topic", "/sensing/lidar/scan")
        self.declare_parameter("odometry.topic", "/localization/kinematic_state")
        self.declare_parameter("control.topic", "/control/command/control_cmd")
        self.declare_parameter("control.dt", 0.05)
        self.declare_parameter("vehicle.min_speed_mps", 0.0)
        self.declare_parameter("vehicle.max_speed_mps", 4.165)
        self.declare_parameter("vehicle.max_accel_mps2", SIM_ACCEL_LIMIT_MPS2)
        self.declare_parameter("vehicle.max_brake_mps2", 5.0)
        self.declare_parameter("vehicle.max_steer_rad", 0.75)
        self.declare_parameter("debug", False)

        self.debug = bool(self.get_parameter("debug").value)
        self.log_interval = float(self.get_parameter("log_interval_sec").value)
        self.control_dt = float(self.get_parameter("control.dt").value)
        self.min_speed = float(self.get_parameter("vehicle.min_speed_mps").value)
        self.max_speed = float(self.get_parameter("vehicle.max_speed_mps").value)
        self.max_accel = min(float(self.get_parameter("vehicle.max_accel_mps2").value), SIM_ACCEL_LIMIT_MPS2)
        self.max_brake = float(self.get_parameter("vehicle.max_brake_mps2").value)
        self.max_steer = float(self.get_parameter("vehicle.max_steer_rad").value)

        self.range_min = float(self.get_parameter("lidar.range_min").value)
        self.range_max = float(self.get_parameter("lidar.range_max").value)
        self.scan_topic = str(self.get_parameter("scan.topic").value)
        self.odometry_topic = str(self.get_parameter("odometry.topic").value)
        self.control_topic = str(self.get_parameter("control.topic").value)

        model_path_param = str(self.get_parameter("model.path").value)
        if not model_path_param:
            raise ValueError("model.path is empty. Set a PPO policy .npz path before starting the node.")

        model_path = resolve_policy_path(model_path_param)
        self.policy = NumpyPpoPolicy(model_path)
        self.input_dim = self.policy.input_dim
        self.current_speed = 0.0
        self.current_steer = 0.0
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
            f"PPO lidar RL controller ready: scan={self.scan_topic}, odom={self.odometry_topic}, "
            f"control={self.control_topic}, input_dim={self.input_dim}, model={model_path}"
        )

    def odometry_callback(self, msg: Odometry) -> None:
        self.current_speed = float(msg.twist.twist.linear.x)

    def scan_callback(self, msg: LaserScan) -> None:
        start_time = time.monotonic()

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.range_max, posinf=self.range_max, neginf=self.range_max)
        ranges = np.clip(ranges, self.range_min, self.range_max)
        has_vehicle_state = (
            self.input_dim > 2
            and len(ranges) != self.input_dim
            and (len(ranges) == self.input_dim - 2 or len(ranges) % (self.input_dim - 2) == 0)
        )
        lidar_dim = self.input_dim - 2 if has_vehicle_state else self.input_dim
        if len(ranges) != lidar_dim:
            idx = np.linspace(0, max(len(ranges) - 1, 0), lidar_dim, dtype=int)
            ranges = ranges[idx] if len(ranges) else np.full(lidar_dim, self.range_max, dtype=np.float32)
        obs = ranges / self.range_max
        if has_vehicle_state:
            vehicle_state = np.array(
                [
                    np.clip(self.current_speed / max(self.max_speed, 1e-6), 0.0, 1.0),
                    0.5 * (np.clip(self.current_steer / max(self.max_steer, 1e-6), -1.0, 1.0) + 1.0),
                ],
                dtype=np.float32,
            )
            obs = np.concatenate([obs, vehicle_state])
        action = self.policy.predict(obs).reshape(-1)
        if action.size < 1:
            self.get_logger().warn("PPO policy returned no actions; skipping command publication.")
            return

        target_speed = self.max_speed
        speed_error = target_speed - self.current_speed
        accel_limit = self.max_accel if speed_error >= 0.0 else self.max_brake
        accel = float(np.clip(speed_error / max(self.control_dt, 1e-3), -accel_limit, accel_limit))
        steer = float(np.clip(float(action[-1]) * self.max_steer, -self.max_steer, self.max_steer))
        self.current_steer = steer

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
    node = LidarRlControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
