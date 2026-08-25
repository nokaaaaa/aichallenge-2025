// Copyright 2023 Tier IV, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <rclcpp/rclcpp.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

using Trajectory = autoware_auto_planning_msgs::msg::Trajectory;
using TrajectoryPoint = autoware_auto_planning_msgs::msg::TrajectoryPoint;

class CSVToTrajectory : public rclcpp::Node
{
public:
  CSVToTrajectory() : Node("csv_to_trajectory_node")
  {
    const auto rb_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
    pub_ = this->create_publisher<Trajectory>("trajectory", rb_qos);
    set_parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&CSVToTrajectory::on_parameter_event, this, std::placeholders::_1));


    declare_parameter("csv_path", "");
    interpolation_factor_ = declare_parameter<int>("interpolation_factor", 5);
    z_= declare_parameter<float>("z");
    std::string csv_path = get_parameter("csv_path").as_string();
    
    if (csv_path.empty()) {
      RCLCPP_ERROR(get_logger(), "CSV path is not specified");
      return;
    }
    
    if (!loadCSVTrajectory(csv_path)) {
      RCLCPP_ERROR(get_logger(), "Failed to load CSV file: %s", csv_path.c_str());
      return;
    }
    current_csv_path_ = csv_path;
    
    RCLCPP_INFO(
      get_logger(), "Loaded trajectory from CSV with %zu points",
      csv_trajectory_.points.size());

    timer_ = rclcpp::create_timer(
      this, get_clock(), std::chrono::seconds(1),
      std::bind(&CSVToTrajectory::publish_trajectory, this));

  }

private:
  bool loadCSVTrajectory(const std::string & csv_path)
  {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
      return false;
    }
    
    std::string line;
    std::getline(file, line);
    
    csv_trajectory_.header.stamp = this->now();
    csv_trajectory_.header.frame_id = "map";

    std::vector<TrajectoryPoint> loaded_points;
    
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
      
      loaded_points.push_back(point);
    }
    
    csv_trajectory_.points.clear();
    const auto interpolated_points = interpolateTrajectory(loaded_points);
    for (const auto & point : interpolated_points) {
      csv_trajectory_.points.push_back(point);
    }
    return !csv_trajectory_.points.empty();
  }

  std::vector<TrajectoryPoint> interpolateTrajectory(
    const std::vector<TrajectoryPoint> & points) const
  {
    if (points.size() < 2 || interpolation_factor_ <= 1) {
      return points;
    }

    const int factor = std::max(1, interpolation_factor_);
    std::vector<TrajectoryPoint> interpolated;
    interpolated.reserve(points.size() * factor);

    for (size_t i = 0; i < points.size(); ++i) {
      const auto & start = points[i];
      const auto & end = points[(i + 1) % points.size()];
      for (int step = 0; step < factor; ++step) {
        const double ratio = static_cast<double>(step) / static_cast<double>(factor);
        interpolated.push_back(interpolatePoint(start, end, ratio));
      }
    }

    return interpolated;
  }

  TrajectoryPoint interpolatePoint(
    const TrajectoryPoint & start, const TrajectoryPoint & end, const double ratio) const
  {
    TrajectoryPoint point = start;
    point.pose.position.x = lerp(start.pose.position.x, end.pose.position.x, ratio);
    point.pose.position.y = lerp(start.pose.position.y, end.pose.position.y, ratio);
    point.pose.position.z = z_;
    point.pose.orientation = interpolateYaw(start.pose.orientation, end.pose.orientation, ratio);
    point.longitudinal_velocity_mps = lerp(
      start.longitudinal_velocity_mps, end.longitudinal_velocity_mps, ratio);
    point.lateral_velocity_mps = lerp(
      start.lateral_velocity_mps, end.lateral_velocity_mps, ratio);
    point.acceleration_mps2 = lerp(start.acceleration_mps2, end.acceleration_mps2, ratio);
    point.heading_rate_rps = lerp(start.heading_rate_rps, end.heading_rate_rps, ratio);
    return point;
  }

  static double lerp(const double start, const double end, const double ratio)
  {
    return start + (end - start) * ratio;
  }

  static geometry_msgs::msg::Quaternion interpolateYaw(
    const geometry_msgs::msg::Quaternion & start,
    const geometry_msgs::msg::Quaternion & end,
    const double ratio)
  {
    const double start_yaw = yawFromQuaternion(start);
    const double end_yaw = yawFromQuaternion(end);
    const double yaw =
      start_yaw + normalizeAngle(end_yaw - start_yaw) * ratio;
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(yaw * 0.5);
    q.w = std::cos(yaw * 0.5);
    return q;
  }

  static double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
  {
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  static double normalizeAngle(const double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }
  
  void publish_trajectory()
  {
    if (csv_trajectory_.points.empty()) {
      RCLCPP_WARN(get_logger(), "No trajectory points to publish");
      return;
    }
    
    csv_trajectory_.header.stamp = this->now();
    pub_->publish(csv_trajectory_);
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(), 60000 /*ms*/, "Published trajectory with %zu points", csv_trajectory_.points.size());
  }

  rcl_interfaces::msg::SetParametersResult on_parameter_event(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "";

    for (const auto & param : parameters) {
      if (param.get_name() == "csv_path") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
          std::string new_csv_path = param.as_string();
          // new_csv_pathがFileSystemのパスであることを確認
          if (!std::filesystem::exists(new_csv_path)) {
            RCLCPP_ERROR(get_logger(), "File does not exist: '%s'", new_csv_path.c_str());
            result.successful = false;
            result.reason = "File does not exist.";
            continue;
          }

          if (new_csv_path != current_csv_path_) {
            RCLCPP_INFO(get_logger(), "csv_path parameter changed from '%s' to '%s'", 
                        current_csv_path_.c_str(), new_csv_path.c_str());
            
            // 新しいCSVファイルの読み込みを試みる
            if (loadCSVTrajectory(new_csv_path)) {
              current_csv_path_ = new_csv_path;
              RCLCPP_INFO(get_logger(), "Successfully loaded new trajectory from CSV: %s with %zu points", 
                          current_csv_path_.c_str(), csv_trajectory_.points.size());
            } else {
              RCLCPP_ERROR(get_logger(), "Failed to load new CSV file: %s. Keeping old trajectory.", new_csv_path.c_str());
              result.successful = false;
              result.reason = "Failed to load new CSV file.";
            }
          }
        } else {
          RCLCPP_WARN(get_logger(), "Parameter 'csv_path' received with wrong type. Expected string.");
          result.successful = false;
          result.reason = "Invalid type for csv_path parameter.";
        }
      } else if (param.get_name() == "z") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE || param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
          z_ = static_cast<float>(param.as_double());
          RCLCPP_INFO(get_logger(), "z parameter changed to %f", z_);
        } else {
          RCLCPP_WARN(get_logger(), "Parameter 'z' received with wrong type. Expected float/double.");
          result.successful = false;
          result.reason = "Invalid type for z parameter.";
        }
      } else if (param.get_name() == "interpolation_factor") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
          const int new_factor = param.as_int();
          if (new_factor < 1) {
            RCLCPP_WARN(get_logger(), "Parameter 'interpolation_factor' must be >= 1.");
            result.successful = false;
            result.reason = "Invalid interpolation_factor.";
            continue;
          }
          interpolation_factor_ = new_factor;
          if (!current_csv_path_.empty() && loadCSVTrajectory(current_csv_path_)) {
            RCLCPP_INFO(
              get_logger(), "interpolation_factor changed to %d, trajectory now has %zu points",
              interpolation_factor_, csv_trajectory_.points.size());
          }
        } else {
          RCLCPP_WARN(
            get_logger(), "Parameter 'interpolation_factor' received with wrong type. Expected integer.");
          result.successful = false;
          result.reason = "Invalid type for interpolation_factor parameter.";
        }
      }
    }
    return result;
  }
  
  rclcpp::Publisher<Trajectory>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  Trajectory csv_trajectory_;
  float z_;
  int interpolation_factor_;
  std::string current_csv_path_;
  OnSetParametersCallbackHandle::SharedPtr set_parameter_callback_handle_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CSVToTrajectory>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
