#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "moveit/move_group_interface/move_group_interface.h"
#include "moveit_msgs/msg/robot_trajectory.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Quaternion.h"

#include "ur_assist/srv/move_to_pose.hpp"

using namespace std::chrono_literals;

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
      "go_to_tf",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto logger = node->get_logger();

  // Параметры
  const std::string planning_group =
      node->declare_parameter<std::string>("planning_group", "ur_manipulator");
  const std::string base_frame =
      node->declare_parameter<std::string>("base_frame", "base_link");
  const std::string tool0_frame =
      node->declare_parameter<std::string>("tool0_frame", "tool0");
  const std::string tcp_frame =
      node->declare_parameter<std::string>("tcp_frame", "tool0_controller");

  RCLCPP_INFO(
      logger,
      "go_to_tf: planning_group='%s', base_frame='%s', tool0_frame='%s', tcp_frame='%s'",
      planning_group.c_str(), base_frame.c_str(), tool0_frame.c_str(), tcp_frame.c_str());

  // TF2
  auto tf_buffer   = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // MoveIt
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group = std::make_shared<MoveGroupInterface>(node, planning_group);

  RCLCPP_INFO(logger, "Planning frame: %s",
              move_group->getPlanningFrame().c_str());
  RCLCPP_INFO(logger, "End effector link (current): %s",
              move_group->getEndEffectorLink().c_str());

  move_group->setPoseReferenceFrame(base_frame);
  move_group->setEndEffectorLink(tool0_frame);
  move_group->setMaxVelocityScalingFactor(0.2);
  move_group->setMaxAccelerationScalingFactor(0.2);

  // Функция ожидания TF
  auto getTransform =
      [node, tf_buffer, logger](const std::string& target, const std::string& source)
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

  using ur_assist::srv::MoveToPose;

  // Сервис: целевая поза TCP (tool0_controller) в base_frame
  auto service = node->create_service<MoveToPose>(
    "go_to_tf",
    [&, node, tf_buffer, move_group, logger, base_frame, tool0_frame, tcp_frame]
    (const std::shared_ptr<MoveToPose::Request> request,
     std::shared_ptr<MoveToPose::Response> response)
    {
      geometry_msgs::msg::Pose target_tcp_pose_msg = request->target;

      RCLCPP_INFO(logger,
                  "Service /go_to_tf called, moving TCP '%s' to given pose in '%s'",
                  tcp_frame.c_str(), base_frame.c_str());

      // 1. TF: текущие base->tool0 и base->tcp_frame
      geometry_msgs::msg::TransformStamped tf_base_tool0_msg;
      geometry_msgs::msg::TransformStamped tf_base_tcp_current_msg;

      try
      {
        tf_base_tool0_msg = getTransform(base_frame, tool0_frame);
        tf_base_tcp_current_msg = getTransform(base_frame, tcp_frame);
      }
      catch (const std::exception& ex)
      {
        RCLCPP_ERROR(logger, "TF lookup failed: %s", ex.what());
        response->success = false;
        response->message = std::string("TF lookup failed: ") + ex.what();
        return;
      }

      tf2::Transform T_base_tool0;
      tf2::Transform T_base_tcp_current;
      tf2::Transform T_base_tcp_target;

      tf2::fromMsg(tf_base_tool0_msg.transform, T_base_tool0);
      tf2::fromMsg(tf_base_tcp_current_msg.transform, T_base_tcp_current);
      tf2::fromMsg(target_tcp_pose_msg, T_base_tcp_target);  // Pose -> Transform

      // 2. Постоянный сдвиг tool0 -> TCP:
      //    T_tool0_tcp = T_base_tool0^{-1} * T_base_tcp_current
      tf2::Transform T_tool0_tcp = T_base_tool0.inverse() * T_base_tcp_current;

      // 3. Линейный путь TCP от текущей позы к целевой
      tf2::Vector3 p_start = T_base_tcp_current.getOrigin();
      tf2::Vector3 p_goal  = T_base_tcp_target.getOrigin();
      tf2::Quaternion q_start = T_base_tcp_current.getRotation();
      tf2::Quaternion q_goal  = T_base_tcp_target.getRotation();

      double distance = (p_goal - p_start).length();
      const double cartesian_step = 0.01;  // 1 см
      int steps = std::max(10, static_cast<int>(distance / cartesian_step));

      std::vector<geometry_msgs::msg::Pose> waypoints;
      waypoints.reserve(steps + 1);

      for (int i = 0; i <= steps; ++i)
      {
        double s = static_cast<double>(i) / static_cast<double>(steps);  // [0,1]

        tf2::Vector3 p = p_start * (1.0 - s) + p_goal * s;
        tf2::Quaternion q = q_start.slerp(q_goal, s);
        q.normalize();

        tf2::Transform T_base_tcp_i(q, p);

        // Переводим в позу tool0:
        // T_base_tool0_i = T_base_tcp_i * T_tool0_tcp^{-1}
        tf2::Transform T_base_tool0_i = T_base_tcp_i * T_tool0_tcp.inverse();

        geometry_msgs::msg::Pose pose_msg;
        pose_msg.position.x = T_base_tool0_i.getOrigin().x();
        pose_msg.position.y = T_base_tool0_i.getOrigin().y();
        pose_msg.position.z = T_base_tool0_i.getOrigin().z();
        pose_msg.orientation = tf2::toMsg(T_base_tool0_i.getRotation());

        waypoints.push_back(pose_msg);
      }

      // 4. Cartesian-путь в пространстве tool0
      moveit_msgs::msg::RobotTrajectory trajectory;
      const double jump_threshold = 0.0;
      const double eef_step = cartesian_step;

      move_group->setStartStateToCurrentState();

      double fraction = move_group->computeCartesianPath(
        waypoints, eef_step, jump_threshold, trajectory);

      RCLCPP_INFO(logger,
                  "Cartesian path fraction: %.1f%%", fraction * 100.0);

      if (fraction < 0.99)
      {
        RCLCPP_WARN(logger,
                    "Cartesian path not fully achievable (%.1f%%). Aborting.",
                    fraction * 100.0);
        response->success = false;
        response->message = "Cartesian path fraction < 0.99";
        return;
      }

      // 5. Исполнение траектории
      auto exec_result = move_group->execute(trajectory);

      if (exec_result != moveit::core::MoveItErrorCode::SUCCESS)
      {
        RCLCPP_ERROR(logger,
                     "Failed to execute Cartesian trajectory, error code: %d",
                     exec_result.val);
        response->success = false;
        response->message = "Failed to execute Cartesian trajectory";
        return;
      }

      RCLCPP_INFO(logger,
                  "Successfully executed Cartesian path for TCP '%s'",
                  tcp_frame.c_str());

      response->success = true;
      response->message = "OK";
    });

  RCLCPP_INFO(logger, "Service /go_to_tf is ready");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
