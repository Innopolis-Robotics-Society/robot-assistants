cmake_minimum_required(VERSION 3.8)
project(iros_object_scanner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(PCL REQUIRED COMPONENTS common filters segmentation search kdtree)

add_executable(sphere_pose_generator_node
  src/sphere_pose_generator_node.cpp
)

ament_target_dependencies(sphere_pose_generator_node
  rclcpp
  geometry_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
)


add_executable(cloud_shot src/cloud_shot.cpp)
ament_target_dependencies(cloud_shot
  rclcpp
  sensor_msgs
  std_srvs
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
)

target_link_libraries(cloud_shot
  ${PCL_LIBRARIES}
)


install(TARGETS
  sphere_pose_generator_node
  cloud_shot
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
