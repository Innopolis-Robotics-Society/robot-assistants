cmake_minimum_required(VERSION 3.8)
project(iros_assistant_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- find dependencies ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ur_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(iros_custom_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)


# --- install launch/rviz/config ---
install(DIRECTORY
  launch
  rviz
  config
  DESTINATION share/${PROJECT_NAME}
)

# --- exec_ik_move executable (внутри реализован сервис MoveToPose) ---
add_executable(exec_ik_move src/exec_ik_move.cpp)

ament_target_dependencies(exec_ik_move
  rclcpp
  geometry_msgs
  std_srvs
  tf2_ros
  tf2_geometry_msgs
  moveit_core
  iros_custom_msgs
  moveit_ros_planning_interface
)

# --- go_to_frame executable (внутри реализован сервис MoveToPose) ---
add_executable(go_to_frame src/go_to_frame.cpp)

ament_target_dependencies(go_to_frame
  rclcpp
  geometry_msgs
  std_srvs
  tf2_ros
  tf2_geometry_msgs
  moveit_core
  iros_custom_msgs
  moveit_ros_planning_interface
)


# --- gripper_control executable (внутри реализован сервис MoveToPose) ---
add_executable(gripper_control src/gripper_control.cpp)

ament_target_dependencies(gripper_control
  rclcpp
  geometry_msgs
  std_srvs
  ur_msgs
  tf2_ros
  tf2_geometry_msgs
  moveit_core
  iros_custom_msgs
  moveit_ros_planning_interface
)

# --- install executables ---
install(TARGETS 
  exec_ik_move
  go_to_frame
  gripper_control
  DESTINATION lib/${PROJECT_NAME}
)

# --- tests ---
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
