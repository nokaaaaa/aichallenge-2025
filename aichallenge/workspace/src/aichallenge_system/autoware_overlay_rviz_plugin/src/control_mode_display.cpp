#include "control_mode_display.hpp"

#include <QPainter>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_rendering/render_system.hpp>

namespace autoware_overlay_rviz_plugin
{
ControlModeDisplay::ControlModeDisplay()
{
  width_property_ = new rviz_common::properties::IntProperty(
    "Width", 130, "Width of the mode overlay", this, SLOT(updateOverlaySize()));
  height_property_ = new rviz_common::properties::IntProperty(
    "Height", 80, "Height of the mode overlay", this, SLOT(updateOverlaySize()));
  left_property_ = new rviz_common::properties::IntProperty(
    "Center X", 340, "Horizontal center offset", this, SLOT(updateOverlayPosition()));
  top_property_ = new rviz_common::properties::IntProperty(
    "Top", 10, "Top position of the overlay", this, SLOT(updateOverlayPosition()));
}

void ControlModeDisplay::onInitialize()
{
  rviz_common::Display::onInitialize();
  rviz_rendering::RenderSystem::get()->prepareOverlays(scene_manager_);
  static int count = 0;
  overlay_ = std::make_shared<OverlayObject>("ControlModeDisplayObject" + std::to_string(count++));
  updateOverlaySize();
  updateOverlayPosition();
  overlay_->show();

  auto ros_node = context_->getRosNodeAbstraction();
  topic_property_ = std::make_unique<rviz_common::properties::RosTopicProperty>(
    "Control Mode Topic", "/control/mode", "std_msgs/msg/String", "Active control mode", this,
    SLOT(topicUpdated()));
  topic_property_->initialize(ros_node);
}

ControlModeDisplay::~ControlModeDisplay()
{
  subscription_.reset();
  topic_property_.reset();
  overlay_.reset();
}

void ControlModeDisplay::update(float, float)
{
  if (!overlay_) return;
  auto buffer = overlay_->getBuffer();
  auto hud = buffer.getQImage(*overlay_);
  hud.fill(Qt::transparent);
  drawWidget(hud);
}

void ControlModeDisplay::drawWidget(QImage & hud)
{
  std::lock_guard<std::mutex> lock(mutex_);
  QPainter painter(&hud);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.setBrush(QColor(0, 0, 0, 210));
  painter.setPen(QColor(120, 120, 120));
  painter.drawRoundedRect(QRectF(1, 1, hud.width() - 2, hud.height() - 2), 10, 10);

  painter.setPen(QColor(210, 210, 210));
  painter.setFont(QFont("Quicksand", 10, QFont::DemiBold));
  painter.drawText(QRectF(0, 10, hud.width(), 18), Qt::AlignCenter, "MODE");

  QColor color(0, 230, 120);
  if (mode_ == "overtake") color = QColor(255, 196, 0);
  if (mode_ == "recovery") color = QColor(255, 80, 80);
  painter.setPen(color);
  painter.setFont(QFont("Quicksand", 15, QFont::Bold));
  painter.drawText(QRectF(3, 32, hud.width() - 6, 30), Qt::AlignCenter, QString::fromStdString(mode_));
}

void ControlModeDisplay::onEnable()
{
  if (overlay_) overlay_->show();
  topicUpdated();
}

void ControlModeDisplay::onDisable()
{
  subscription_.reset();
  if (overlay_) overlay_->hide();
}

void ControlModeDisplay::reset()
{
  rviz_common::Display::reset();
  mode_ = "free";
  if (overlay_) overlay_->hide();
}

void ControlModeDisplay::updateOverlaySize()
{
  if (!overlay_) return;
  overlay_->updateTextureSize(width_property_->getInt(), height_property_->getInt());
  overlay_->setDimensions(overlay_->getTextureWidth(), overlay_->getTextureHeight());
  queueRender();
}

void ControlModeDisplay::updateOverlayPosition()
{
  if (!overlay_) return;
  overlay_->setPosition(
    left_property_->getInt(), top_property_->getInt(), HorizontalAlignment::CENTER,
    VerticalAlignment::TOP);
  queueRender();
}

void ControlModeDisplay::topicUpdated()
{
  subscription_.reset();
  auto ros_node = context_->getRosNodeAbstraction().lock();
  subscription_ = ros_node->get_raw_node()->create_subscription<std_msgs::msg::String>(
    topic_property_->getTopicStd(), rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(),
    [this](const std_msgs::msg::String::SharedPtr msg) { updateMode(msg); });
}

void ControlModeDisplay::updateMode(const std_msgs::msg::String::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  mode_ = msg->data;
  queueRender();
}
}  // namespace autoware_overlay_rviz_plugin

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(autoware_overlay_rviz_plugin::ControlModeDisplay, rviz_common::Display)
