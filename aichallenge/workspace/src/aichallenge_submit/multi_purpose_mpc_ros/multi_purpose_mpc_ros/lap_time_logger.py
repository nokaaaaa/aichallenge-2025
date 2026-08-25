#!/usr/bin/env python3

import csv
import os
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from multi_purpose_mpc_ros.core.utils import format_time


class LapTimeLogger(Node):
    def __init__(self) -> None:
        super().__init__("lap_time_logger")

        self.declare_parameter("status_topic", "/awsim/status")
        self.declare_parameter("output_dir", ".")
        self.declare_parameter("lap_time_file", "lap_times.csv")
        self.declare_parameter("summary_file", "lap_summary.txt")

        status_topic = str(self.get_parameter("status_topic").value)
        output_dir = os.path.abspath(str(self.get_parameter("output_dir").value))
        lap_time_file = str(self.get_parameter("lap_time_file").value)
        self._summary_file = str(self.get_parameter("summary_file").value)

        os.makedirs(output_dir, exist_ok=True)
        self._lap_time_log_path = os.path.join(output_dir, lap_time_file)
        self._summary_path = os.path.join(output_dir, self._summary_file)

        self._current_laps: Optional[int] = None
        self._last_lap_time = 0.0
        self._lap_times = []

        with open(self._lap_time_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lap", "lap_time_sec", "lap_time_text", "elapsed_time_sec"])

        self._status_sub = self.create_subscription(
            Float32MultiArray, status_topic, self._status_callback, 1)
        self.get_logger().info(f"Lap time log: {self._lap_time_log_path}")

    def _status_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 3:
            return

        laps = int(msg.data[1])
        lap_time = float(msg.data[2])

        if self._current_laps is None:
            self._current_laps = 1 if laps == 0 else laps

        if laps > self._current_laps:
            completed_lap = self._current_laps
            completed_time = self._last_lap_time
            self._lap_times.append(completed_time)
            self._append_lap_time(completed_lap, completed_time)
            self._write_summary()
            self.get_logger().info(
                f"Lap {completed_lap} completed. Lap time: {completed_time:.3f} s")
            self._current_laps = laps

        self._last_lap_time = lap_time

    def _append_lap_time(self, lap: int, lap_time: float) -> None:
        elapsed_time = sum(self._lap_times)
        with open(self._lap_time_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([lap, f"{lap_time:.6f}", format_time(lap_time), f"{elapsed_time:.6f}"])

    def _write_summary(self) -> None:
        if not self._lap_times:
            return

        total_time = sum(self._lap_times)
        average_lap_time = total_time / len(self._lap_times)
        fastest_lap_time = min(self._lap_times)

        with open(self._summary_path, "w") as f:
            f.write(f"total_laps: {len(self._lap_times)}\n")
            f.write(f"total_time_sec: {total_time:.6f}\n")
            f.write(f"total_time: {format_time(total_time)}\n")
            f.write(f"average_lap_time_sec: {average_lap_time:.6f}\n")
            f.write(f"average_lap_time: {format_time(average_lap_time)}\n")
            f.write(f"fastest_lap_time_sec: {fastest_lap_time:.6f}\n")
            f.write(f"fastest_lap_time: {format_time(fastest_lap_time)}\n")
            for i, lap_time in enumerate(self._lap_times, start=1):
                f.write(f"lap_{i}_time_sec: {lap_time:.6f}\n")
                f.write(f"lap_{i}_time: {format_time(lap_time)}\n")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LapTimeLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
