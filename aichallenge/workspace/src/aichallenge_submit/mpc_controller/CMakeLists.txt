cmake_minimum_required(VERSION 3.8)
project(mpc_controller)

# 必要なビルドシステム
find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# osqp-eigen をサブディレクトリとして追加
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../osqp-eigen
  ${CMAKE_CURRENT_BINARY_DIR}/osqp-eigen-build
)

# 実行ファイルを定義（ターゲット名は mpc_controller）
ament_auto_add_executable(mpc_controller
  src/mpc_controller.cpp
)

# osqp-eigen のヘッダディレクトリをインクルードに追加
target_include_directories(mpc_controller PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../osqp-eigen/include
)

# osqp-eigen ライブラリをリンク（正しい名前空間）
target_link_libraries(mpc_controller OsqpEigen::OsqpEigen)

# パッケージとして完結させる
ament_auto_package()
