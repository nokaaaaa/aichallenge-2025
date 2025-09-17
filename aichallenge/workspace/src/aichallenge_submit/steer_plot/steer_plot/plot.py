#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import SteeringReport

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import deque


class SteeringPlotNode(Node):
    def __init__(self):
        super().__init__("steering_plot_node")

        # 購読
        self.sub_cmd = self.create_subscription(
            AckermannControlCommand,
            "output/control_cmd",
            self.cmd_callback,
            10
        )
        self.sub_report = self.create_subscription(
            SteeringReport,
            "/vehicle/status/steering_status",
            self.report_callback,
            10
        )

        # データ保存用（最新N秒分）
        self.buffer_time = deque(maxlen=500)   # 時間
        self.buffer_cmd = deque(maxlen=500)    # コマンド値
        self.buffer_report = deque(maxlen=500) # 実際値
        self.start_time = time.time()

        # グラフ設定
        plt.style.use("seaborn")  # seaborn-v0_8 → seaborn に変更
        self.fig, self.ax = plt.subplots()
        self.line_cmd, = self.ax.plot([], [], label="Cmd Steering")
        self.line_report, = self.ax.plot([], [], label="Report Steering")
        self.ax.legend()
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Steering Angle [rad]")

        # アニメーション開始
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=100
        )

    def cmd_callback(self, msg: AckermannControlCommand):
        t = time.time() - self.start_time
        self.buffer_time.append(t)
        self.buffer_cmd.append(msg.lateral.steering_tire_angle)

    def report_callback(self, msg: SteeringReport):
        t = time.time() - self.start_time
        self.buffer_time.append(t)
        self.buffer_report.append(msg.steering_tire_angle)
        print(f"Steering Report: {msg.steering_tire_angle:.3f} rad")

    def update_plot(self, frame):
        if len(self.buffer_time) == 0:
            return self.line_cmd, self.line_report

        self.line_cmd.set_data(self.buffer_time, self.buffer_cmd)
        self.line_report.set_data(self.buffer_time, self.buffer_report)

        # 時間軸を左に流す（最後の5秒分だけ表示）
        t_max = self.buffer_time[-1]
        self.ax.set_xlim(max(0, t_max - 5), t_max)
        self.ax.set_ylim(-1.0, 1.0)  # ここは車両の最大舵角に合わせて調整

        return self.line_cmd, self.line_report

    def spin(self):
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = SteeringPlotNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
