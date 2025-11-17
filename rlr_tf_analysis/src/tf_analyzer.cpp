#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <memory>

class TFAnalyzer : public rclcpp::Node {
public:
  TFAnalyzer() : Node("tf_analyzer") {
    // Parameters
    this->declare_parameter("camera_frame", "camera");
    this->declare_parameter("base_frame", "base_link");
    this->declare_parameter("max_age", 1.0);
    this->declare_parameter("publish_markers", true);
    this->declare_parameter("publish_corrected_tf", true);  // Новый параметр
    
    camera_frame_ = this->get_parameter("camera_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    max_age_ = this->get_parameter("max_age").as_double();
    publish_markers_ = this->get_parameter("publish_markers").as_bool();
    publish_corrected_tf_ = this->get_parameter("publish_corrected_tf").as_bool();

    // TF setup
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // Инициализация TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    
    // Subscribers and publishers
    detection_sub_ = this->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>(
      "/apriltag/detections", 10,
      std::bind(&TFAnalyzer::tag_detection_callback, this, std::placeholders::_1));
    
    if(publish_markers_) {
      marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/tag_visualization", 10);
    }
    
    RCLCPP_INFO(this->get_logger(), "TF Analyzer initialized");
    RCLCPP_INFO(this->get_logger(), "Camera frame: %s", camera_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "Base frame: %s", base_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "Publish corrected TF: %s", 
                publish_corrected_tf_ ? "true" : "false");
  }

private:
  void tag_detection_callback(const apriltag_msgs::msg::AprilTagDetectionArray::SharedPtr msg) {
    for(const auto & detection : msg->detections) {
      process_tag(detection);
    }
    
    if(publish_markers_) {
      publish_visualization();
    }
  }

  void process_tag(const apriltag_msgs::msg::AprilTagDetection & detection) {
    try {
      std::string tag_frame = "marker" + std::to_string(detection.id);
      
      // Получаем трансформацию base_link -> markerX
      geometry_msgs::msg::TransformStamped transform = 
        tf_buffer_->lookupTransform(base_frame_, tag_frame, tf2::TimePointZero);
      
      // Сохраняем для визуализации
      tag_transforms_[detection.id] = transform;
      
      // Публикуем исправленную трансформацию
      if(publish_corrected_tf_) {
        geometry_msgs::msg::TransformStamped corrected_transform;
        
        // Заполняем заголовок
        corrected_transform.header.stamp = this->now();
        corrected_transform.header.frame_id = base_frame_;
        corrected_transform.child_frame_id = "corrected_marker_" + std::to_string(detection.id);
        
        // Копируем трансформацию (здесь можно добавить коррекцию)
        corrected_transform.transform = transform.transform;
        
        // Пример коррекции: смещение на 5 см по Z
        // corrected_transform.transform.translation.z += 0.05;
        
        // Публикуем
        tf_broadcaster_->sendTransform(corrected_transform);
      }
      
      // Логирование позиции
      double x = transform.transform.translation.x;
      double y = transform.transform.translation.y;
      double z = transform.transform.translation.z;
      
      RCLCPP_INFO(this->get_logger(), "Tag %d position: (%.2f, %.2f, %.2f) relative to %s", 
                  detection.id, x, y, z, base_frame_.c_str());
                
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "TF exception: %s", ex.what());
    }
  }

  void publish_visualization() {
    visualization_msgs::msg::MarkerArray marker_array;
    rclcpp::Time now = this->now();
    
    // Remove old markers
    visualization_msgs::msg::Marker delete_markers;
    delete_markers.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_markers);
    
    // Create new markers
    for(const auto & [id, transform] : tag_transforms_) {
      // Skip old transforms
      if((now - rclcpp::Time(transform.header.stamp)).seconds() > max_age_) continue;
      
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = base_frame_;
      marker.header.stamp = now;
      marker.ns = "apriltags";
      marker.id = id;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      
      marker.pose.position.x = transform.transform.translation.x;
      marker.pose.position.y = transform.transform.translation.y;
      marker.pose.position.z = transform.transform.translation.z;
      marker.pose.orientation = transform.transform.rotation;
      
      marker.scale.x = 0.05;
      marker.scale.y = 0.05;
      marker.scale.z = 0.01;
      
      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
      marker.color.a = 0.7f;
      
      marker.lifetime = rclcpp::Duration::from_seconds(max_age_);
      
      marker_array.markers.push_back(marker);
    }
    
    if (marker_pub_) {
      marker_pub_->publish(marker_array);
    }
  }

  // Member variables
  rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr detection_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;  // Для публикации исправленных TF
  
  std::map<int, geometry_msgs::msg::TransformStamped> tag_transforms_;
  std::string camera_frame_;
  std::string base_frame_;
  double max_age_;
  bool publish_markers_;
  bool publish_corrected_tf_;  // Флаг для включения/выключения публикации
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TFAnalyzer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}