#include "mpc_controller/mpc_controller.hpp"
#include <OsqpEigen/OsqpEigen.h> 
#include <tf2/utils.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

using namespace std;

double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

MPCControllerNode::MPCControllerNode(const rclcpp::NodeOptions & options)
: Node("mpc_controller_node", options) {
    declare_parameters();
    load_parameters();
    if (!load_path()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load path file. Shutting down.");
        rclcpp::shutdown();
        return;
    }
    const auto bv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
    sub_kinematics_ = create_subscription<Odometry>(
    "input/kinematics", bv_qos, [this](const Odometry::SharedPtr msg) { odometry_ = msg; });
    sub_velocity_ = create_subscription<VelocityReport>(
    "/vehicle/status/velocity_status", bv_qos, [this](const VelocityReport::SharedPtr msg) { velocity_report_ = msg; });
    sub_steering_ = create_subscription<SteeringReport>(
    "/vehicle/status/steering_status", bv_qos, [this](const SteeringReport::SharedPtr msg) { steering_report_ = msg; });
    
    pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);
    pub_pred_path_ = this->create_publisher<visualization_msgs::msg::Marker>("output/predicted_path", 1);

    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(dt_ * 1000)),
        std::bind(&MPCControllerNode::control_timer_callback, this));
    current_state_.setZero();
    current_steer_ = 0.0;
    current_v_ = 0.0;

    RCLCPP_INFO(this->get_logger(), "MPC Controller Node has been initialized.");
}

void MPCControllerNode::declare_parameters() {
    this->declare_parameter<double>("L", 1.087);
    this->declare_parameter<double>("tau", 0.0001);
    this->declare_parameter<double>("dt", 0.01);
    this->declare_parameter<int>("N", 20);
    this->declare_parameter<std::string>("path_csv_file", "");
    this->declare_parameter<std::vector<double>>("Q_weights", {2.0, 8.0, 0.0});
    this->declare_parameter<double>("R_weight", 10.0);
    this->declare_parameter<double>("max_steer_angle", 1);
    this->declare_parameter<double>("min_steer_angle", -1);
    this->declare_parameter<double>("lowpass_steer", 1.0); 
    this->declare_parameter<double>("kp", 100.0); 
}

void MPCControllerNode::load_parameters() {
    this->get_parameter("L", L_);
    this->get_parameter("tau", tau_);
    this->get_parameter("dt", dt_);
    this->get_parameter("N", N_);
    this->get_parameter("path_csv_file", path_csv_file_);
    
    std::vector<double> q_weights;
    this->get_parameter("Q_weights", q_weights);
    Q_ = Eigen::Matrix3d::Zero();
    Q_(0, 0) = q_weights[0]; // y
    Q_(1, 1) = q_weights[1]; // theta
    Q_(2, 2) = q_weights[2]; // delta

    double r_weight;
    this->get_parameter("R_weight", r_weight);
    R_(0,0) = r_weight;
    
    this->get_parameter("max_steer_angle", max_steer_angle_);
    this->get_parameter("min_steer_angle", min_steer_angle_);
    this->get_parameter("lowpass_steer", lowpass_steer_);
    this->get_parameter("kp", kp_);
}

bool MPCControllerNode::load_path() {
    std::ifstream file(path_csv_file_);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Cannot open path file: %s", path_csv_file_.c_str());
        return false;
    }

    reference_path_.clear();
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.find("s_m;") == 0) {
            continue;
        }

        std::stringstream ss(line);
        std::string item;
        Waypoint wp;

        try {
            std::getline(ss, item, ';'); 
            std::getline(ss, item, ';'); wp.x = std::stod(item);
            std::getline(ss, item, ';'); wp.y = std::stod(item);
            std::getline(ss, item, ';'); wp.psi = std::stod(item);
            std::getline(ss, item, ';'); wp.kappa = std::stod(item);
            std::getline(ss, item, ';'); wp.v = std::stod(item);
            reference_path_.push_back(wp);
        } catch (const std::invalid_argument& e) {
            RCLCPP_WARN(this->get_logger(), "Could not parse line in CSV: %s", line.c_str());
        }
    }

    file.close();
    RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints.", reference_path_.size());
    return !reference_path_.empty();
}


size_t MPCControllerNode::find_closest_waypoint(double x, double y) {
    double min_dist_sq = std::numeric_limits<double>::max();
    size_t found_idx = closest_idx_;
/*
    if(init == false) {
        // 初回は全探索
        start_idx_ = 0;
        end_idx_ = reference_path_.size();
        init = true;
    } else {
        // 前回の最近傍点からの探索を行う
        start_idx_ = std::max(0, static_cast<int>(closest_idx_) - 5);
        end_idx_ = std::min(start_idx_ + 10, reference_path_.size());
    }
*/  
     start_idx_ = 0;
        end_idx_ = reference_path_.size();

    for (size_t i = start_idx_; i < end_idx_; ++i) {
        double dx = x - reference_path_[i].x;
        double dy = y - reference_path_[i].y;
        double dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            found_idx = i;
        }
    }
    return found_idx;
}

void MPCControllerNode::control_timer_callback() {
if (!odometry_) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000, "Waiting for odometry topic...");
    return;
}

if (!velocity_report_) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000, "Waiting for velocity_report topic...");
    return;
}

if (!steering_report_) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000, "Waiting for steering_report topic...");
    return;
}

if (reference_path_.empty()) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000, "Reference path is empty. Waiting for path...");
    return;
}
    // --- 1. 現在の車両状態を取得 ---
    double current_vehicle_x = odometry_->pose.pose.position.x; 
    double current_vehicle_y = odometry_->pose.pose.position.y; 
    double current_vehicle_yaw = tf2::getYaw(odometry_->pose.pose.orientation);

    current_v_ = velocity_report_->longitudinal_velocity;
    current_steer_ = steering_report_->steering_tire_angle;

    size_t offset = 1;

    // --- 2. 最も近いウェイポイントを探索し、状態量を計算 ---
    size_t closest_idx = find_closest_waypoint(current_vehicle_x, current_vehicle_y);
    closest_idx += offset; // オフセットを加える
    
    // 経路の終端に近づいた場合の処理
    if (closest_idx >= reference_path_.size()) {
        closest_idx -= reference_path_.size();
    }
    RCLCPP_WARN_THROTTLE(
    this->get_logger(), *this->get_clock(), 5000, "Closest waypoint index: %zu", closest_idx);
    const Waypoint& ref_wp = reference_path_[closest_idx];
    double path_yaw = normalize_angle(ref_wp.psi+M_PI/2); 
    
    // 座標系をパスに合わせる
    double dx = current_vehicle_x - ref_wp.x;
    double dy = current_vehicle_y - ref_wp.y;

    // 現在の状態量 [横方向誤差, 方位角誤差, 現在のステアリング角]
    //ワールド座標をπ/2 - psi 回転することで 経路近傍の座標系に
    current_state_[0] = -dx * sin(path_yaw) + dy * cos(path_yaw);         // lateral_error (y)
    current_state_[1] = normalize_angle(current_vehicle_yaw - path_yaw); // heading_error (theta)
    current_state_[2] = current_steer_;                             // current_steer (delta)

    // --- 3. 予測モデルの線形化 ---
    int nx = 3; // 状態量 [y, th, d] の数
    int nu = 1; // 入力量 [目標ステアリング角 u] の数

    vector<Eigen::MatrixXd> A_horizon(N_);
    vector<Eigen::MatrixXd> B_horizon(N_);
    vector<Eigen::VectorXd> w_horizon(N_);

    for (int i = 0; i < N_; ++i) {
        size_t path_index = closest_idx +i;
        if (path_index >= reference_path_.size()) {
            path_index -= reference_path_.size(); 
        }
        const auto& wp = reference_path_[path_index];

        const auto& wp_next = reference_path_[(path_index + 1) % reference_path_.size()];


        //差分を計算
        double delta_x = wp.x - wp_next.x;
        double delta_y = wp.y - wp_next.y;
        double delta_yaw = normalize_angle( wp.psi - wp_next.psi); //正規化しないと2π超えておかしなことになる(この値を三角関数に使わないことがあるため)

        // next waypoint 座標系における wpの座標
        double px = delta_x * cos(wp_next.psi + M_PI/2) + delta_y * sin(wp_next.psi + M_PI/2);
        double py = -delta_x * sin(wp_next.psi+ M_PI/2) + delta_y * cos(wp_next.psi+ M_PI/2);

        // next waypoint 座標系における x軸と接線のなす角
      //  double psi_1 = wp.psi + M_PI/2;
       // double psi_2 = wp_next.psi + M_PI/2;

        // 三角関数の計算
        double cos_d = cos(delta_yaw);
        double sin_d = sin(delta_yaw);
        
        // current waypointにおける目標ステアリング角
        double delta_ref = atan(L_ * wp.kappa);

        double R ;

        //旋回半径
        if (abs (wp.kappa) < 0.001) {
            R = 1000.0; // ほぼ直線の場合は大きな半径を設定
        } else {
            R = 1.0 / wp.kappa; // 半径
        }


        // sqrt_f
        double sqrt_f ;
        double f;
        f = cos_d*cos_d + 2 * px * sin_d / R - px*px / (R*R);

        if(f < 0) sqrt_f = 0.0; 
        else sqrt_f = sqrt(f);


        //y_k+1における y_kの係数
        double y_coeff_y = sin_d*sin_d / sqrt_f -px * sin_d / (R* sqrt_f) + cos_d;

        //y_k+1における theta_kの係数
        double theta_coeff_y = R * sin_d * cos_d / sqrt_f -px * cos_d / sqrt_f -R*sin_d;
    
        //y_k+1の定数項

        double y_const = -R * sqrt_f + py + R * cos_d;

        //theta_k+1における y_kの係数
        double y_coeff_theta = sin_d *(tan(delta_ref)-delta_ref/(cos(delta_ref)*cos(delta_ref)))/ L_;
        
        //theta_k+1における theta_kの係数
        double delta_coeff_theta = -px/(L_ * cos(delta_ref) * cos(delta_ref));

        //theta_k+1の定数項
        double theta_const =  delta_yaw -px*(tan(delta_ref) - delta_ref/(cos(delta_ref) * cos(delta_ref)))/L_;
   
        //ホライズンとwaypointを対応させるためにvdtをwaypoint間の距離にする
         Eigen::MatrixXd A_k(nx, nx);
        A_k << y_coeff_y, theta_coeff_y, 0,
               y_coeff_theta , 1, delta_coeff_theta,
               0, 0, 1- dt_tau;

        Eigen::MatrixXd B_k(nx, nu);
        B_k << 0, 0, dt_tau;

        Eigen::VectorXd w_k(nx);
        w_k << y_const,
              theta_const, 
              0;

        A_horizon[i] = A_k;
        B_horizon[i] = B_k;
        w_horizon[i] = w_k;
    }


    // --- 4. 予測モデルの拡張 ---
    //X=F * x_0 + G * U + S * W
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(N_ * nx, nx);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(N_ * nx, N_ * nu);
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(N_ * nx, N_ * nx);
    Eigen::VectorXd W = Eigen::VectorXd::Zero(N_ * nx);

    Eigen::MatrixXd A_prod = Eigen::MatrixXd::Identity(nx, nx);
    for (int i = 0; i < N_; ++i) {
        A_prod = A_horizon[i] * A_prod;
        F.block(i * nx, 0, nx, nx) = A_prod;

        W.block(i*nx, 0, nx, 1) = w_horizon[i];

        for (int j = 0; j < N_; ++j) {
            if (j > i) continue;
            Eigen::MatrixXd A_prod_temp = Eigen::MatrixXd::Identity(nx, nx);
            for(int k = j + 1; k <= i; ++k) {
                A_prod_temp = A_horizon[k] * A_prod_temp;
            }
            if (j == i) {
                G.block(i * nx, i * nu, nx, nu) = B_horizon[i];
                S.block(i * nx, i * nx, nx, nx) = Eigen::MatrixXd::Identity(nx, nx);
            } else {
                G.block(i * nx, j * nu, nx, nu) = A_prod_temp * B_horizon[j];
                S.block(i * nx, j * nx, nx, nx) = A_prod_temp;
            }
        }
    }

        // --- 参照入力ベクトル U_ref の作成 ---
    Eigen::VectorXd U_ref = Eigen::VectorXd::Zero(N_ * nu);
    for (int i = 0; i < N_; ++i) {
        size_t path_index = closest_idx + i;
        if (path_index >= reference_path_.size()) {
            path_index -= reference_path_.size(); 
        }
        const auto& wp = reference_path_[path_index];

        // 各予測ステップでの参照ステアリング角を計算
        double delta_ref = atan(L_ * wp.kappa);
        U_ref(i) = delta_ref;
    }


    // --- 4. QPの定式化 (コスト関数と制約) ---
    Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(N_ * nx, N_ * nx);
    Eigen::MatrixXd R_bar = Eigen::MatrixXd::Zero(N_ * nu, N_ * nu);

    for(int i=0; i<N_; ++i){
        Q_bar.block(i*nx, i*nx, nx, nx) = Q_;
        R_bar.block(i*nu, i*nu, nu, nu) = R_;
    }

    // コスト関数: J = 0.5 * U^T * H * U + g^T * U
    
    Eigen::MatrixXd H = G.transpose() * Q_bar * G + R_bar;
    Eigen::VectorXd g = G.transpose() * Q_bar * (F * current_state_ + S * W) - R_bar * U_ref;

    // --- 5. QPソルバーによる最適化 ---
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    solver.data()->setNumberOfVariables(N_ * nu);
    // 制約の数を更新
    solver.data()->setNumberOfConstraints(N_ * nu); 

    Eigen::SparseMatrix<double> H_sparse = H.sparseView();
    solver.data()->setHessianMatrix(H_sparse);
    solver.data()->setGradient(g);
    

     //y制約ないときのみ
    Eigen::VectorXd lower_bound_combined = Eigen::VectorXd::Constant(N_ * nu, min_steer_angle_);
    Eigen::VectorXd upper_bound_combined = Eigen::VectorXd::Constant(N_ * nu, max_steer_angle_);
    Eigen::SparseMatrix<double> A_constraints(N_ * nu, N_ * nu);

    // 更新した制約行列と上下限ベクトルをセット
    solver.data()->setLinearConstraintsMatrix(A_constraints);
    solver.data()->setLowerBound(lower_bound_combined);
    solver.data()->setUpperBound(upper_bound_combined);

    if (!solver.initSolver()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize OSQP solver.");
        return;
    }
    
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
        RCLCPP_WARN(this->get_logger(), "Failed to solve QP problem.");
        return;
    }

    // --- 6. 最適解を取得し、制御指令をパブリッシュ ---
    Eigen::VectorXd U_star = solver.getSolution();



    // 将来の状態量を計算 X = F*x_current + G*U* + S*W
    Eigen::VectorXd X_pred = F * current_state_ + G * U_star + S * W;

     // --- 7. 予測ホライズンを可視化 ---
    visualization_msgs::msg::Marker predicted_path_marker;
    predicted_path_marker.header.frame_id = "map"; 
    predicted_path_marker.header.stamp = this->get_clock()->now();
    predicted_path_marker.ns = "mpc_prediction";
    predicted_path_marker.id = 0;
    predicted_path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    predicted_path_marker.action = visualization_msgs::msg::Marker::ADD;
    predicted_path_marker.pose.orientation.w = 1.0;
    predicted_path_marker.scale.x = 0.05;
    predicted_path_marker.color.g = 1.0;
    predicted_path_marker.color.a = 1.0;
    
    for (int i = 0; i < N_; ++i) {
        // この予測ステップに対応する参照ウェイポイント
        size_t path_index = closest_idx + i;
        if (path_index >= reference_path_.size()) {
            path_index -= reference_path_.size();
        }
        const auto& ref_wp = reference_path_[path_index];
        double y_err = X_pred(i * nx); 
        double path_yaw = normalize_angle(ref_wp.psi + M_PI / 2.0);
        double predicted_x = ref_wp.x - y_err * sin(path_yaw);
        double predicted_y = ref_wp.y + y_err * cos(path_yaw);
        geometry_msgs::msg::Point p;
        p.x = predicted_x;
        p.y = predicted_y;
        p.z = odometry_->pose.pose.position.z; 
        predicted_path_marker.points.push_back(p);
    }
    pub_pred_path_->publish(predicted_path_marker);
    double optimal_steer_cmd = U_star(0);

    optimal_steer_cmd = std::clamp(optimal_steer_cmd, min_steer_angle_, max_steer_angle_);

    AckermannControlCommand cmd;
    cmd.stamp = this->get_clock()->now();
    cmd.lateral.steering_tire_angle = optimal_steer_cmd;
    cmd.longitudinal.acceleration = kp_ * (ref_wp.v - current_v_);
    //車庫脱出
    if(current_v_ <3.0) cmd.lateral.steering_tire_angle = 0.1;
    else if(current_v_ <4.5) cmd.lateral.steering_tire_angle = -0.05;
    else if(current_v_ <5.5) cmd.lateral.steering_tire_angle = -0.2;
    pub_cmd_->publish(cmd);

}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<MPCControllerNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}