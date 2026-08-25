#include "simple_pure_pursuit/simple_pure_pursuit.hpp"

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <iterator>

namespace simple_pure_pursuit
{

using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcYawDeviation;

SimplePurePursuit::SimplePurePursuit()
: Node("simple_pure_pursuit"),
  // initialize parameters
  wheel_base_(declare_parameter<float>("wheel_base", 2.14)),
  lookahead_gain_(declare_parameter<float>("lookahead_gain", 1.0)),
  lookahead_min_distance_(declare_parameter<float>("lookahead_min_distance", 1.0)),
  speed_proportional_gain_(declare_parameter<float>("speed_proportional_gain", 1.0)),
  use_external_target_vel_(declare_parameter<bool>("use_external_target_vel", false)),
  external_target_vel_(declare_parameter<float>("external_target_vel", 0.0)),
  steering_tire_angle_gain_(declare_parameter<float>("steering_tire_angle_gain", 1.0)),
  steering_tire_angle_lim_(declare_parameter<float>("steering_tire_angle_lim", 0.64)),
  control_delay_sec_(declare_parameter<float>("control_delay_sec", 0.0)),
  max_acceleration_(declare_parameter<float>("max_acceleration", 3.0))
{
  pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);
  pub_raw_cmd_ = create_publisher<AckermannControlCommand>("output/raw_control_cmd", 1);
  pub_lookahead_point_ = create_publisher<PointStamped>("/control/debug/lookahead_point", 1);

  const auto bv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
  sub_kinematics_ = create_subscription<Odometry>(
    "input/kinematics", bv_qos, [this](const Odometry::SharedPtr msg) { odometry_ = msg; });
  sub_trajectory_ = create_subscription<Trajectory>(
    "input/trajectory", bv_qos, [this](const Trajectory::SharedPtr msg) { trajectory_ = msg; });

  using namespace std::literals::chrono_literals;
  timer_ = create_wall_timer(10ms, std::bind(&SimplePurePursuit::onTimer, this));
}

AckermannControlCommand zeroAckermannControlCommand(rclcpp::Time stamp)
{
  AckermannControlCommand cmd;
  cmd.stamp = stamp;
  cmd.longitudinal.stamp = stamp;
  cmd.longitudinal.speed = 0.0;
  cmd.longitudinal.acceleration = 0.0;
  cmd.lateral.stamp = stamp;
  cmd.lateral.steering_tire_angle = 0.0;
  return cmd;
}

void SimplePurePursuit::onTimer()
{
  // check data
  if (!subscribeMessageAvailable()) {
    return;
  }

  const double current_yaw = tf2::getYaw(odometry_->pose.pose.orientation);
  const double current_longitudinal_vel = odometry_->twist.twist.linear.x;
  const double current_yaw_rate = odometry_->twist.twist.angular.z;

  geometry_msgs::msg::Point control_position = odometry_->pose.pose.position;
  double control_yaw = current_yaw;
  if (control_delay_sec_ > 0.0) {
    control_position.x += current_longitudinal_vel * std::cos(current_yaw) * control_delay_sec_;
    control_position.y += current_longitudinal_vel * std::sin(current_yaw) * control_delay_sec_;
    control_yaw += current_yaw_rate * control_delay_sec_;
  }

  size_t closet_traj_point_idx = findNearestIndex(trajectory_->points, control_position);

  // publish zero command
  AckermannControlCommand cmd = zeroAckermannControlCommand(get_clock()->now());

  // get closest trajectory point from current position
  TrajectoryPoint closet_traj_point = trajectory_->points.at(closet_traj_point_idx);

  // calc longitudinal speed and acceleration
  double target_longitudinal_vel =
    use_external_target_vel_ ? external_target_vel_ : closet_traj_point.longitudinal_velocity_mps;
  cmd.longitudinal.speed = target_longitudinal_vel;
  cmd.longitudinal.acceleration = 3.0;

  // calc lateral control
  //// calc lookahead distance
  double lookahead_distance = lookahead_gain_ * target_longitudinal_vel + lookahead_min_distance_;
  //// calc center coordinate of rear wheel
  double rear_x = control_position.x - wheel_base_ / 2.0 * std::cos(control_yaw);
  double rear_y = control_position.y - wheel_base_ / 2.0 * std::sin(control_yaw);
  //// search lookahead point
  auto lookahead_point_itr = std::find_if(
    trajectory_->points.begin() + closet_traj_point_idx, trajectory_->points.end(),
    [&](const TrajectoryPoint & point) {
      return std::hypot(point.pose.position.x - rear_x, point.pose.position.y - rear_y) >=
             lookahead_distance;
    });
  if (lookahead_point_itr == trajectory_->points.end()) {
    lookahead_point_itr = std::prev(trajectory_->points.end());
  }
  double lookahead_point_x = lookahead_point_itr->pose.position.x;
  double lookahead_point_y = lookahead_point_itr->pose.position.y;

  geometry_msgs::msg::PointStamped lookahead_point_msg;
  lookahead_point_msg.header.stamp = get_clock()->now();
  lookahead_point_msg.header.frame_id = "map";
  lookahead_point_msg.point.x = lookahead_point_x;
  lookahead_point_msg.point.y = lookahead_point_y;
  lookahead_point_msg.point.z = closet_traj_point.pose.position.z;
  pub_lookahead_point_->publish(lookahead_point_msg);

  // calc steering angle for lateral control
  double alpha =
    std::atan2(lookahead_point_y - rear_y, lookahead_point_x - rear_x) - control_yaw;
  alpha = std::atan2(std::sin(alpha), std::cos(alpha));
  const double raw_steering_tire_angle =
    std::atan2(2.0 * wheel_base_ * std::sin(alpha), lookahead_distance);
  cmd.lateral.steering_tire_angle = std::clamp(
    steering_tire_angle_gain_ * raw_steering_tire_angle,
    -steering_tire_angle_lim_,
    steering_tire_angle_lim_);

  pub_cmd_->publish(cmd);
  cmd.lateral.steering_tire_angle = raw_steering_tire_angle;
  pub_raw_cmd_->publish(cmd);
}

bool SimplePurePursuit::subscribeMessageAvailable()
{
  if (!odometry_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "odometry is not available");
    return false;
  }
  if (!trajectory_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "trajectory is not available");
    return false;
  }
  if (trajectory_->points.empty()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/,  "trajectory points is empty");
      return false;
    }
  return true;
}
}  // namespace simple_pure_pursuit

int main(int argc, char const * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_pure_pursuit::SimplePurePursuit>());
  rclcpp::shutdown();
  return 0;
}
