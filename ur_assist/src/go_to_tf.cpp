#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include "ur_assist/srv/move_to_pose.hpp"

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);

  // Узел с авто-декларацией параметров (важно для MoveIt2)
  auto node = std::make_shared<rclcpp::Node>(
      "ur10e_move_server",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto logger = node->get_logger();

  // Планировочная группа UR10e (проверь в SRDF/RViz, чаще всего "ur_manipulator")
  static const std::string PLANNING_GROUP = "ur_manipulator";
  using moveit::planning_interface::MoveGroupInterface;

  auto move_group = std::make_shared<MoveGroupInterface>(node, PLANNING_GROUP);

  RCLCPP_INFO(logger, "Planning frame: %s",
              move_group->getPlanningFrame().c_str());
  RCLCPP_INFO(logger, "End effector link: %s",
              move_group->getEndEffectorLink().c_str());

  // Базовые настройки планировщика (как в hello_moveit_ur10e)
  move_group->setPlannerId("RRTConnectkConfigDefault");   // должен быть в ompl_planning.yaml
  move_group->setNumPlanningAttempts(10);                 // выбрать более короткий путь
  move_group->setPlanningTime(2.0);
  move_group->setMaxVelocityScalingFactor(0.2);
  move_group->setMaxAccelerationScalingFactor(0.2);
  move_group->setGoalPositionTolerance(0.001);      // 1 мм
  move_group->setGoalOrientationTolerance(0.005);   // ~0.3°

  // Создаём сервис /move_to_pose
  auto service = node->create_service<ur_assist::srv::MoveToPose>(
    "move_to_pose",
    [node, move_group](const std::shared_ptr<ur_assist::srv::MoveToPose::Request> request,
                       std::shared_ptr<ur_assist::srv::MoveToPose::Response> response)
    {
      auto logger = node->get_logger();
      const auto& target = request->target;

      RCLCPP_INFO(logger,
                  "Received target pose: position(%.3f, %.3f, %.3f)",
                  target.position.x, target.position.y, target.position.z);

      // Старт — текущее состояние робота
      move_group->setStartStateToCurrentState();

      // Цель — поза TCP
      move_group->setPoseTarget(target);

      MoveGroupInterface::Plan plan;
      bool success = static_cast<bool>(move_group->plan(plan));

      if (!success)
      {
        RCLCPP_ERROR(logger, "Planning failed");
        response->success = false;
        response->message = "Planning failed";
        return;
      }

      RCLCPP_INFO(logger, "Planning succeeded, executing trajectory...");

      auto exec_result = move_group->execute(plan);
      if (!static_cast<bool>(exec_result))
      {
        RCLCPP_ERROR(logger, "Execution failed");
        response->success = false;
        response->message = "Execution failed";
        return;
      }

      RCLCPP_INFO(logger, "Motion finished successfully");
      response->success = true;
      response->message = "OK";
    });

  RCLCPP_INFO(logger, "Service /move_to_pose is ready");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
