#include <rclcpp/rclcpp.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <std_msgs/msg/bool.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using Trajectory = autoware_auto_planning_msgs::msg::Trajectory;
using TrajectoryPoint = autoware_auto_planning_msgs::msg::TrajectoryPoint;
using BoolMsg = std_msgs::msg::Bool;

class CSVToTrajectory : public rclcpp::Node
{
public:
  CSVToTrajectory() : Node("csv_to_trajectory_node"), current_trajectory_ptr_(nullptr)
  {
    // Publisher
    const auto rb_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
    pub_ = this->create_publisher<Trajectory>("trajectory", rb_qos);

    // Subscriber
    change_sub_ = this->create_subscription<BoolMsg>(
      "/change", 10, std::bind(&CSVToTrajectory::on_change_request, this, std::placeholders::_1));

    // Parameters
    declare_parameter("csv_path1", "");
    declare_parameter("csv_path2", "");
    z_ = declare_parameter<float>("z", 0.0);

    std::string csv_path1 = get_parameter("csv_path1").as_string();
    std::string csv_path2 = get_parameter("csv_path2").as_string();

    if (csv_path1.empty() || csv_path2.empty()) {
      RCLCPP_ERROR(get_logger(), "Both csv_path1 and csv_path2 must be specified.");
      return;
    }

    if (!loadCSVTrajectory(csv_path1, trajectory1_)) {
      RCLCPP_ERROR(get_logger(), "Failed to load CSV file for path1: %s", csv_path1.c_str());
      return;
    }
    RCLCPP_INFO(get_logger(), "Loaded trajectory1 with %zu points", trajectory1_.points.size());

    if (!loadCSVTrajectory(csv_path2, trajectory2_)) {
      RCLCPP_ERROR(get_logger(), "Failed to load CSV file for path2: %s", csv_path2.c_str());
      return;
    }
    RCLCPP_INFO(get_logger(), "Loaded trajectory2 with %zu points", trajectory2_.points.size());

    // 初期状態は path1
    current_trajectory_ptr_ = &trajectory1_;
    RCLCPP_INFO(get_logger(), "Initially publishing trajectory1: %s", csv_path1.c_str());

    // 起動時刻を記録
    start_time_ = this->now();

    // Timer
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),  // 10Hz
      std::bind(&CSVToTrajectory::publish_trajectory, this));
  }

private:
  bool changed_ = false;  
  rclcpp::Time start_time_;  // ノード起動時刻

  bool loadCSVTrajectory(const std::string & csv_path, Trajectory & trajectory_out)
  {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
      return false;
    }

    trajectory_out.points.clear();
    trajectory_out.header.stamp = this->now();
    trajectory_out.header.frame_id = "map";

    std::string line;
    std::getline(file, line);  // skip header

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string token;
      std::vector<double> values;

      while (std::getline(ss, token, ',')) {
        values.push_back(std::stod(token));
      }

      if (values.size() != 8) {
        RCLCPP_WARN(get_logger(), "Invalid CSV line format, expected 8 values");
        continue;
      }

      TrajectoryPoint point;
      point.pose.position.x = values[0];
      point.pose.position.y = values[1];
      point.pose.position.z = z_;

      point.pose.orientation.x = values[3];
      point.pose.orientation.y = values[4];
      point.pose.orientation.z = values[5];
      point.pose.orientation.w = values[6];

      point.longitudinal_velocity_mps = values[7];
      point.lateral_velocity_mps = 0.0;
      point.acceleration_mps2 = 0.0;
      point.heading_rate_rps = 0.0;

      trajectory_out.points.push_back(point);
    }

    return !trajectory_out.points.empty();
  }

  void on_change_request(const BoolMsg::ConstSharedPtr msg)
  {
    // まだ切り替えていない && 5秒以上経過している && msgがtrue
    if (!changed_ && msg->data) {
      auto elapsed = this->now() - start_time_;
      if (elapsed.seconds() >= 5.0) {
        current_trajectory_ptr_ = &trajectory2_;
        changed_ = true;
        RCLCPP_INFO(get_logger(), "Switched to trajectory2 (after %.2f sec).", elapsed.seconds());
      } else {
        RCLCPP_WARN(get_logger(), "Change request received before 5 sec (%.2f sec). Ignored.", elapsed.seconds());
      }
    }
  }

  void publish_trajectory()
  {
    if (!current_trajectory_ptr_ || current_trajectory_ptr_->points.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Current trajectory is not set or is empty. Not publishing.");
      return;
    }

    current_trajectory_ptr_->header.stamp = this->now();
    pub_->publish(*current_trajectory_ptr_);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 60000,
      "Published trajectory with %zu points", current_trajectory_ptr_->points.size());
  }

  rclcpp::Publisher<Trajectory>::SharedPtr pub_;
  rclcpp::Subscription<BoolMsg>::SharedPtr change_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  Trajectory trajectory1_;
  Trajectory trajectory2_;
  Trajectory* current_trajectory_ptr_;

  float z_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CSVToTrajectory>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
