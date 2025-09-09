#ifndef SIMPLE_PURE_PURSUIT_HPP_
#define SIMPLE_PURE_PURSUIT_HPP_

#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <autoware_auto_vehicle_msgs/msg/velocity_report.hpp>
#include <autoware_auto_vehicle_msgs/msg/steering_report.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <fstream>

namespace simple_pure_pursuit {

using autoware_auto_control_msgs::msg::AckermannControlCommand;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PointStamped;
using geometry_msgs::msg::Twist;
using nav_msgs::msg::Odometry;
using autoware_auto_vehicle_msgs::msg::VelocityReport;
using autoware_auto_vehicle_msgs::msg::SteeringReport;
using std_msgs::msg::Float32MultiArray;
using std_msgs::msg::Bool;



class SimplePurePursuit : public rclcpp::Node {
 public:
  explicit SimplePurePursuit();
  
  // subscribers
  rclcpp::Subscription<Odometry>::SharedPtr sub_kinematics_;
  rclcpp::Subscription<Trajectory>::SharedPtr sub_trajectory_;
  rclcpp::Subscription<VelocityReport>::SharedPtr sub_velocity_;
  rclcpp::Subscription<SteeringReport>::SharedPtr sub_steering_;
  rclcpp::Subscription<Float32MultiArray>::SharedPtr sub_status_;
  
  
  // publishers
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_cmd_;
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_raw_cmd_;
  rclcpp::Publisher<PointStamped>::SharedPtr pub_lookahead_point_;  
  rclcpp::Publisher<Bool>::SharedPtr change_publisher_;

  // timer
  rclcpp::TimerBase::SharedPtr timer_;

  // updated by subscribers
  Trajectory::SharedPtr trajectory_;
  Odometry::SharedPtr odometry_;
  VelocityReport::ConstSharedPtr velocity_report_;
  SteeringReport::ConstSharedPtr steering_report_;
  Float32MultiArray::SharedPtr status_;



  // pure pursuit parameters
  const double wheel_base_;
  const double tau_;
  const double lookahead_gain_;
  const double lookahead_min_distance_;
  const double speed_proportional_gain_;
  const bool use_external_target_vel_;
  const double external_target_vel_;
  const double steering_tire_angle_gain_;


 private:
  void onTimer();
  bool subscribeMessageAvailable();
  void on_velocity_report(const VelocityReport::ConstSharedPtr msg);
 
  double last_steer_ = 0.0;
  bool fixed_steer = false;
  double current_steer_ = 0.0;

  bool steer_timer_started_ = false;
  std::chrono::steady_clock::time_point steer_start_time_;

  //log
  std::ofstream ofs_;

  // (lap, section) -> start time
  std::map<std::pair<int,int>, double> section_start_time_;
  std::map<int, std::map<int, double>> lap_section_times_;

  int prev_lap_ = -1;
  int prev_section_ = -1;
  const int max_sections_ = 8;

};

}  // namespace simple_pure_pursuit

#endif  // SIMPLE_PURE_PURSUIT_HPP_
