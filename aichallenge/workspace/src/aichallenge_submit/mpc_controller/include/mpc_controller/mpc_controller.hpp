#ifndef MPC_CONTROLLER__MPC_CONTROLLER_NODE_HPP_
#define MPC_CONTROLLER__MPC_CONTROLLER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_vehicle_msgs/msg/velocity_report.hpp>
#include <autoware_auto_vehicle_msgs/msg/steering_report.hpp>
#include "visualization_msgs/msg/marker.hpp" 
#include <Eigen/Dense>
#include <vector>
#include <string>

struct Waypoint {
    double x;      // x座標 (m)
    double y;      // y座標 (m)
    double psi;    // ヨー角 (rad)
    double kappa;  // 曲率 (1/m)
    double v ;      // 速度 (m/s)
};

using autoware_auto_control_msgs::msg::AckermannControlCommand;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::Twist;
using nav_msgs::msg::Odometry;
using autoware_auto_vehicle_msgs::msg::VelocityReport;
using autoware_auto_vehicle_msgs::msg::SteeringReport;

class MPCControllerNode : public rclcpp::Node {
public:
    // コンストラクタ
    explicit MPCControllerNode(const rclcpp::NodeOptions & options);

private:
    // Subscription
    //車速 m/s
    rclcpp::Subscription<VelocityReport>::SharedPtr sub_velocity_;
    //ステアリング角 rad
    rclcpp::Subscription<SteeringReport>::SharedPtr sub_steering_;
    //オドメトリ
    rclcpp::Subscription<Odometry>::SharedPtr sub_kinematics_;

    //Publisher
    //アッカーマンコマンド
    rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_cmd_;
    //予測経路
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_pred_path_;

    rclcpp::TimerBase::SharedPtr control_timer_;

    // updated by subscribers
    Odometry::SharedPtr odometry_;
    VelocityReport::ConstSharedPtr velocity_report_;
    SteeringReport::ConstSharedPtr steering_report_;

    // 車両とMPCのパラメータ
    double L_;      // ホイールベース
    double tau_;    // ステアリング遅延時定数
    double dt_;     // 制御周期
    int N_;         // 予測ホライズン
    std::string path_csv_file_;

    // QPの重み行列
    Eigen::Matrix3d Q_;
    Eigen::Matrix<double, 1, 1> R_;

    // 制約
    double max_steer_angle_, min_steer_angle_;

    // 状態量
    double current_v_;
    Eigen::Vector3d current_state_; // 状態ベクトル [y_err, theta_err, delta]
    double current_steer_;

    double lowpass_steer_;
    double last_steer_ = 0.0; 
    double kp_ ; //加速度ゲイン
    double dt_tau = 0.1;
    
    // 経路情報
    std::vector<Waypoint> reference_path_;
    size_t closest_idx_ = 0;
    size_t start_idx_ = 0;
    size_t end_idx_ = 0;
    bool init = false; //最初の探索では全探索

    void declare_parameters();
    void load_parameters();
    bool load_path();
    void control_timer_callback();
    size_t find_closest_waypoint(double x, double y) ;

};

#endif  // MPC_CONTROLLER__MPC_CONTROLLER_NODE_HPP_