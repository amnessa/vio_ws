// Copyright 2018 Open Source Robotics Foundation, Inc.
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

#include <memory>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

/**
 * @brief A ROS2 node that subscribes to pose messages and broadcasts them as TF transforms.
 *
 * This node acts as a bridge between custom pose topics and the TF2 transform tree,
 * allowing other nodes to access pose information through the standard TF2 interface.
 */
class PoseTfBroadcaster : public rclcpp::Node {
public:
  PoseTfBroadcaster()
  : rclcpp::Node("pose_tf_broadcaster")
  {
    // Create transform broadcasters for dynamic and static transforms
    tf_br_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    static_br_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    // Configure QoS settings for TF-like behavior:
    // - KeepLast(100): Keep up to 100 messages in history
    // - best_effort(): Use best-effort reliability (no guaranteed delivery)
    // - durability_volatile(): Don't store messages for late-joining subscribers
    auto tf_qos = rclcpp::QoS(rclcpp::KeepLast(100))
        .best_effort()
        .durability_volatile();

    // Subscribe to dynamic pose updates and broadcast them as TF transforms
    // These transforms are expected to change over time (e.g., robot odometry)
    sub_pose_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "pose", tf_qos,
      [this](const tf2_msgs::msg::TFMessage::SharedPtr msg) {
        tf_br_->sendTransform(msg->transforms);
      });

    // Subscribe to static pose updates and broadcast them as static TF transforms
    // These transforms are expected to remain constant (e.g., sensor mounting positions)
    sub_pose_static_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "pose_static", tf_qos,
      [this](const tf2_msgs::msg::TFMessage::SharedPtr msg) {
        static_br_->sendTransform(msg->transforms);
      });
  }

private:
  // Broadcaster for dynamic transforms (published to /tf)
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_br_;

  // Broadcaster for static transforms (published to /tf_static with latched behavior)
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_br_;

  // Subscription for incoming dynamic pose messages
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr sub_pose_;

  // Subscription for incoming static pose messages
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr sub_pose_static_;
};

int main(int argc, char **argv)
{
  // Initialize ROS2
  rclcpp::init(argc, argv);

  // Create and spin the node until shutdown is requested
  rclcpp::spin(std::make_shared<PoseTfBroadcaster>());

  // Clean up ROS2 resources
  rclcpp::shutdown();
  return 0;
}