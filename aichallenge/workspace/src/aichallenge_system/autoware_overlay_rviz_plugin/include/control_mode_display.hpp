#ifndef CONTROL_MODE_DISPLAY_HPP_
#define CONTROL_MODE_DISPLAY_HPP_

#include "overlay_utils.hpp"

#include <QImage>
#include <rviz_common/display.hpp>
#include <rviz_common/properties/int_property.hpp>
#include <rviz_common/properties/ros_topic_property.hpp>
#include <std_msgs/msg/string.hpp>

#include <memory>
#include <mutex>
#include <string>

namespace autoware_overlay_rviz_plugin
{
class ControlModeDisplay : public rviz_common::Display
{
  Q_OBJECT
public:
  ControlModeDisplay();
  ~ControlModeDisplay() override;

protected:
  void onInitialize() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;
  void onEnable() override;
  void onDisable() override;

private Q_SLOTS:
  void updateOverlaySize();
  void updateOverlayPosition();
  void topicUpdated();

private:
  void drawWidget(QImage & hud);
  void updateMode(const std_msgs::msg::String::ConstSharedPtr & msg);

  std::mutex mutex_;
  OverlayObject::SharedPtr overlay_;
  rviz_common::properties::IntProperty * width_property_;
  rviz_common::properties::IntProperty * height_property_;
  rviz_common::properties::IntProperty * left_property_;
  rviz_common::properties::IntProperty * top_property_;
  std::unique_ptr<rviz_common::properties::RosTopicProperty> topic_property_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  std::string mode_ = "free";
};
}  // namespace autoware_overlay_rviz_plugin

#endif  // CONTROL_MODE_DISPLAY_HPP_
