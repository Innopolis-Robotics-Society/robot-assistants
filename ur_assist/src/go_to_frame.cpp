#include <memory>
#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "ur_assist/srv/move_to_pose.hpp"
#include "ur_assist/srv/go_to_frame.hpp"

using namespace std::chrono_literals;

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
      "go_to_frame",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto logger = node->get_logger();

  // Параметр базового фрейма (в нём задана целевая поза TCP)
  const std::string base_frame =
      node->declare_parameter<std::string>("base_frame", "base_link");

  RCLCPP_INFO(logger, "go_to_frame: base_frame='%s'", base_frame.c_str());

  // TF
  auto tf_buffer   = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // Клиент к уже существующему сервису go_to_tf (MoveToPose)
  using MoveToPose = ur_assist::srv::MoveToPose;
  auto go_to_tf_client = node->create_client<MoveToPose>("go_to_tf");

  // Вспомогательная функция ожидания TF
  auto getTransform =
      [tf_buffer, logger](const std::string& target, const std::string& source)
      -> geometry_msgs::msg::TransformStamped
  {
    while (rclcpp::ok()) {
      try {
        return tf_buffer->lookupTransform(
            target, source, tf2::TimePointZero);
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(logger,
                    "Waiting for transform %s -> %s: %s",
                    target.c_str(), source.c_str(), ex.what());
        rclcpp::sleep_for(200ms);
      }
    }
    throw std::runtime_error("Shutdown while waiting for transform");
  };

  // Сервис, который принимает имя фрейма и запускает go_to_tf
  using GoToFrame = ur_assist::srv::GoToFrame;
  auto service = node->create_service<GoToFrame>(
    "go_to_frame",
    [&, node, go_to_tf_client, base_frame, getTransform]
    (const std::shared_ptr<GoToFrame::Request> request,
     std::shared_ptr<GoToFrame::Response> response)
    {
      const std::string frame = request->frame;
      RCLCPP_INFO(node->get_logger(),
                  "Service /go_to_frame called, frame='%s'", frame.c_str());

      // 1. Проверяем, что сервис go_to_tf доступен
      if (!go_to_tf_client->wait_for_service(1s)) {
        RCLCPP_ERROR(node->get_logger(),
                     "Service 'go_to_tf' not available");
        response->success = false;
        response->message = "Service 'go_to_tf' not available";
        return;
      }

      // 2. Получаем TF base_frame -> frame
      geometry_msgs::msg::TransformStamped tf_frame;
      try
      {
        tf_frame = getTransform(base_frame, frame);
      }
      catch (const std::exception& ex)
      {
        RCLCPP_ERROR(node->get_logger(),
                     "Failed to get transform %s -> %s: %s",
                     base_frame.c_str(), frame.c_str(), ex.what());
        response->success = false;
        response->message = std::string("TF error: ") + ex.what();
        return;
      }

      // 3. Преобразуем Transform в Pose
      geometry_msgs::msg::Pose target_pose;
      target_pose.position.x = tf_frame.transform.translation.x;
      target_pose.position.y = tf_frame.transform.translation.y;
      target_pose.position.z = tf_frame.transform.translation.z;
      target_pose.orientation = tf_frame.transform.rotation;

      // 4. Готовим запрос к go_to_tf (MoveToPose)
      auto move_req = std::make_shared<MoveToPose::Request>();
      move_req->target = target_pose;

      // 5. Отправляем запрос (fire-and-forget)
      go_to_tf_client->async_send_request(move_req);

      RCLCPP_INFO(node->get_logger(),
                  "Sent request to 'go_to_tf' with pose from frame '%s'", frame.c_str());

      response->success = true;
      response->message = "Request sent to go_to_tf";
    });

  RCLCPP_INFO(logger, "Service /go_to_frame is ready");

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
