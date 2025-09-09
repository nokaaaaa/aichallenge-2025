#include "simple_pure_pursuit/simple_pure_pursuit.hpp"

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2/utils.h>

#include <algorithm>

using namespace std;

namespace simple_pure_pursuit
{

using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcYawDeviation;
using std::placeholders::_1;

SimplePurePursuit::SimplePurePursuit()
: Node("simple_pure_pursuit"),
  // initialize parameters
  wheel_base_(declare_parameter<float>("wheel_base", 2.14)),
  tau_(declare_parameter<double>("tau", 10.0)),
  lookahead_gain_(declare_parameter<float>("lookahead_gain", 1.0)),
  lookahead_min_distance_(declare_parameter<float>("lookahead_min_distance", 1.0)),
  speed_proportional_gain_(declare_parameter<float>("speed_proportional_gain", 1.0)),
  use_external_target_vel_(declare_parameter<bool>("use_external_target_vel", false)),
  external_target_vel_(declare_parameter<float>("external_target_vel", 0.0)),
  steering_tire_angle_gain_(declare_parameter<float>("steering_tire_angle_gain", 1.0))
{
  pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);
  pub_raw_cmd_ = create_publisher<AckermannControlCommand>("output/raw_control_cmd", 1);
  pub_lookahead_point_ = create_publisher<PointStamped>("/control/debug/lookahead_point", 1);
  change_publisher_ = this->create_publisher<Bool>("/change", 10);

  const auto bv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
  sub_kinematics_ = create_subscription<Odometry>(
    "input/kinematics", bv_qos, [this](const Odometry::SharedPtr msg) { odometry_ = msg; });
  sub_trajectory_ = create_subscription<Trajectory>(
    "input/trajectory", bv_qos, [this](const Trajectory::SharedPtr msg) { trajectory_ = msg; });

   sub_velocity_ = create_subscription<VelocityReport>(
    "/vehicle/status/velocity_status", 1, std::bind(&SimplePurePursuit::on_velocity_report, this, _1));
  
  sub_status_ = create_subscription<Float32MultiArray>(
    "/aichallenge/awsim/status", bv_qos, [this](const Float32MultiArray::SharedPtr msg) { status_ = msg; });

      sub_steering_ = create_subscription<SteeringReport>(
    "/vehicle/status/steering_status", bv_qos, [this](const SteeringReport::SharedPtr msg) { steering_report_ = msg; });

    ofs_.open("lap_log.csv", std::ios::out | std::ios::trunc);
    ofs_ << "Lap";
    for (int i = 1; i <= max_sections_; i++) {
      ofs_ << ",Section" << i;
    }
    ofs_ << ",LapTime\n";  

  using namespace std::literals::chrono_literals;
  timer_ =
    rclcpp::create_timer(this, get_clock(), 1ms, std::bind(&SimplePurePursuit::onTimer, this));
}


void SimplePurePursuit::on_velocity_report(const VelocityReport::ConstSharedPtr msg)
{
  velocity_report_ = msg;
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
  size_t closet_traj_point_idx =
    findNearestIndex(trajectory_->points, odometry_->pose.pose.position);

  // publish zero command
  AckermannControlCommand cmd = zeroAckermannControlCommand(get_clock()->now());

  // get closest trajectory point from current position
  TrajectoryPoint closet_traj_point = trajectory_->points.at(closet_traj_point_idx);

  // calc longitudinal speed and acceleration
  // double target_longitudinal_vel =
  //   use_external_target_vel_ ? external_target_vel_ : closet_traj_point.longitudinal_velocity_mps;

   cmd.longitudinal.acceleration =500;

  //時速35km制限
  if(velocity_report_->longitudinal_velocity > 9.8) cmd.longitudinal.acceleration = 0.5;

  // calc lateral control
  //// calc lookahead distance
  //double lookahead_distance = lookahead_gain_ * target_longitudinal_vel + lookahead_min_distance_;
  double lookahead_distance = 9.0;

  //if (closet_traj_point_idx > 75 && closet_traj_point_idx <90) lookahead_distance = 15.0;
  //// calc center coordinate of rear wheel
  double rear_x = odometry_->pose.pose.position.x -
                  wheel_base_ / 2.0 * std::cos(odometry_->pose.pose.orientation.z);
  double rear_y = odometry_->pose.pose.position.y -
                  wheel_base_ / 2.0 * std::sin(odometry_->pose.pose.orientation.z);

  //// search lookahead point
  auto lookahead_point_itr = trajectory_->points.begin() + closet_traj_point_idx;
  size_t total_points = trajectory_->points.size();
  size_t search_count = 0;

  while (search_count < total_points) {
    if (std::hypot(lookahead_point_itr->pose.position.x - rear_x,
                  lookahead_point_itr->pose.position.y - rear_y) >= lookahead_distance) {
      break;
    }
    // 次のポイントに進む（終端に到達したら先頭に戻る）
    lookahead_point_itr++;
    if (lookahead_point_itr == trajectory_->points.end()) {
      lookahead_point_itr = trajectory_->points.begin();
    }
    search_count++;
  }

  // 探索が失敗した場合は最後のポイントを選択
  if (search_count == total_points) {
    lookahead_point_itr = trajectory_->points.end() - 1;
  }

  //特定のインデックスで経路変更
  size_t lookahead_index = std::distance(trajectory_->points.begin(), lookahead_point_itr);

  if(lookahead_index > 33 && lookahead_index <40)
  {
    auto message = std_msgs::msg::Bool();
    message.data = true;
    change_publisher_->publish(message);
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
  double alpha = std::atan2(lookahead_point_y - rear_y, lookahead_point_x - rear_x) -
                 tf2::getYaw(odometry_->pose.pose.orientation);
  
  current_steer_ = steering_report_->steering_tire_angle;

  double target_angle = std::atan2(2.0 * wheel_base_ * std::sin(alpha), lookahead_distance);
  double delta_angle = target_angle - current_steer_;
  cmd.lateral.steering_tire_angle = steering_tire_angle_gain_*(current_steer_ + delta_angle * tau_ );
    cmd.longitudinal.acceleration =500.0;
    //時速35km制限
  if(velocity_report_->longitudinal_velocity > 10.15) cmd.longitudinal.acceleration = 0.0;


  //区間によってsteerの大きくしたり小さくする

  //look_ahead_pointがある区間にあるときは強制的にある点をlook_ahead_pointにする(カーブ後の振動をなくす)

  //ローカル環境で各waypoint(close)にいるときのy偏差を記録して その値も考慮してフィードバックをかける
  // 速度をトリガーに決め打ちで制御

  // if(!steer_timer_started_) cmd.lateral.steering_tire_angle = 0.0;

  // if (velocity_report_->longitudinal_velocity > 3.0)
  // {
  //   if (!steer_timer_started_)
  //   {
  //     steer_timer_started_ = true;
  //     steer_start_time_ = std::chrono::steady_clock::now();
  //   }
  // }
  
  // if (steer_timer_started_)
  // {
  //   auto current_time = std::chrono::steady_clock::now();
  //   double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - steer_start_time_).count();

  //   if (elapsed_seconds <= 1.8)
  //   {
  //     cmd.lateral.steering_tire_angle = 0.05;
  //   }
  //   else if (elapsed_seconds <= 2.6)
  //   {
  //     cmd.lateral.steering_tire_angle = -0.7;
  //   }
  //    else if (elapsed_seconds <= 4.8)
  //   {
  //     cmd.lateral.steering_tire_angle = 0.0;
  //   }

  // }
    
  pub_cmd_->publish(cmd);
  cmd.lateral.steering_tire_angle /=  steering_tire_angle_gain_;
  pub_raw_cmd_->publish(cmd);

  last_steer_ = cmd.lateral.steering_tire_angle;

  // // //セクションのログ
  // if (status_->data.size() < 4) {
  //     RCLCPP_WARN(this->get_logger(), "Message too short");
  //     return;
  //   }
 
  //    int lap = static_cast<int>(status_->data[1]);
  //    double time = this->now().seconds();
  //   int section = static_cast<int>(status_->data[3]);

  //       // lap=0 は無視
  //   if (lap == 0) {
  //     return;
  //   }

  //   auto key = std::make_pair(lap, section);

  //   // セクション開始時刻の登録
  //   if (section_start_time_.find(key) == section_start_time_.end()) {
  //     section_start_time_[key] = time;
  //   }

  //   // セクションが切り替わったら経過時間を計算
  //   if (prev_section_ != -1 && section != prev_section_) {
  //     auto prev_key = std::make_pair(prev_lap_, prev_section_);
  //     if (section_start_time_.find(prev_key) != section_start_time_.end()) {
  //       double start_time = section_start_time_[prev_key];
  //       double elapsed = -time + start_time;

  //       // Lapごとの記録用バッファに格納
  //       lap_section_times_[prev_lap_][prev_section_] = elapsed;

  //       RCLCPP_INFO(this->get_logger(),
  //         "Lap %d Section %d Time: %.3f sec",
  //         prev_lap_, prev_section_, elapsed);

  //       // ---- ラップ切り替わりのタイミングでCSVにまとめて出力 ----
  //       if (lap != prev_lap_ && prev_lap_ > 0) {
  //         ofs_ << prev_lap_;

  //         double lap_time_sum = 0.0;
  //         for (int s = 1; s <= max_sections_; s++) {
  //           if (lap_section_times_[prev_lap_].count(s)) {
  //             double t = lap_section_times_[prev_lap_][s];
  //             ofs_ << "," << t;
  //             lap_time_sum += t;
  //           } else {
  //             ofs_ << ",";
  //           }
  //         }

  //         ofs_ << "," << lap_time_sum << "\n";  
  //         ofs_.flush();

  //         // 過去ラップのデータは不要なら消す
  //         lap_section_times_.erase(prev_lap_);
  //       }
  //     }
  //   }

  //   prev_lap_ = lap;
  //   prev_section_ = section;
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