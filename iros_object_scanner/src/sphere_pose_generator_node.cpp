// iros_object_scanner/src/sphere_pose_generator_node.cpp
#include <rclcpp/rclcpp.hpp>

#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>

namespace
{
constexpr double kEps = 1e-9;

double deg2rad(double d) { return d * M_PI / 180.0; }

tf2::Vector3 vec3_from_param(const std::vector<double>& v, const tf2::Vector3& fallback)
{
  if (v.size() != 3) return fallback;
  return tf2::Vector3(v[0], v[1], v[2]);
}

tf2::Quaternion quat_from_param(const std::vector<double>& q, const tf2::Quaternion& fallback)
{
  if (q.size() != 4) return fallback;
  tf2::Quaternion qq(q[0], q[1], q[2], q[3]);
  if (qq.length2() < kEps) return fallback;
  qq.normalize();
  return qq;
}

tf2::Quaternion quat_from_two_vectors(const tf2::Vector3& from_in, const tf2::Vector3& to_in)
{
  tf2::Vector3 from = from_in;
  tf2::Vector3 to = to_in;
  if (from.length2() < kEps || to.length2() < kEps) {
    return tf2::Quaternion(0, 0, 0, 1);
  }
  from.normalize();
  to.normalize();

  const double dot = std::clamp(from.dot(to), -1.0, 1.0);

  if (dot > 1.0 - 1e-10) {
    return tf2::Quaternion(0, 0, 0, 1);
  }

  if (dot < -1.0 + 1e-10) {
    // 180 deg rotation around any axis orthogonal to "from"
    tf2::Vector3 axis = tf2::Vector3(1, 0, 0).cross(from);
    if (axis.length2() < kEps) axis = tf2::Vector3(0, 1, 0).cross(from);
    axis.normalize();
    tf2::Quaternion q(axis, M_PI);
    q.normalize();
    return q;
  }

  const tf2::Vector3 axis = from.cross(to);
  const double s = std::sqrt((1.0 + dot) * 2.0);
  const double invs = 1.0 / s;

  tf2::Quaternion q(axis.x() * invs, axis.y() * invs, axis.z() * invs, s * 0.5);
  q.normalize();
  return q;
}

tf2::Quaternion compute_lookat_quat(
  const tf2::Vector3& forward_base,              // desired forward in base
  const tf2::Vector3& up_base_in,                // reference up in base
  const tf2::Vector3& look_axis_cam_in,          // forward axis in camera frame
  const tf2::Vector3& up_axis_cam_in)            // up axis in camera frame (to stabilize roll)
{
  tf2::Vector3 f = forward_base;
  if (f.length2() < kEps) return tf2::Quaternion(0, 0, 0, 1);
  f.normalize();

  tf2::Vector3 a_cam = look_axis_cam_in;
  if (a_cam.length2() < kEps) a_cam = tf2::Vector3(0, 0, 1);
  a_cam.normalize();

  tf2::Vector3 u_cam = up_axis_cam_in;
  if (u_cam.length2() < kEps) u_cam = tf2::Vector3(0, -1, 0);
  u_cam.normalize();

  tf2::Vector3 up_base = up_base_in;
  if (up_base.length2() < kEps) up_base = tf2::Vector3(0, 0, 1);
  up_base.normalize();

  // Step 1: rotate camera look axis to desired forward direction
  tf2::Quaternion q1 = quat_from_two_vectors(a_cam, f);

  // Step 2: roll adjustment around f so that projected up aligns to projected up_base
  tf2::Vector3 u1 = tf2::quatRotate(q1, u_cam);

  tf2::Vector3 u1p = u1 - f * (u1.dot(f));
  tf2::Vector3 upp = up_base - f * (up_base.dot(f));

  if (u1p.length2() < 1e-10 || upp.length2() < 1e-10) {
    q1.normalize();
    return q1;
  }

  u1p.normalize();
  upp.normalize();

  const double sinang = f.dot(u1p.cross(upp));
  const double cosang = std::clamp(u1p.dot(upp), -1.0, 1.0);
  const double ang = std::atan2(sinang, cosang);

  tf2::Quaternion qroll(f, ang);
  qroll.normalize();

  tf2::Quaternion q = qroll * q1;
  q.normalize();
  return q;
}

std::string json_string_list(const std::vector<std::string>& xs)
{
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < xs.size(); ++i) {
    ss << "\"" << xs[i] << "\"";
    if (i + 1 < xs.size()) ss << ", ";
  }
  ss << "]";
  return ss.str();
}

std::string make_name(const std::string& prefix, bool append_epoch, int epoch, int digits, int idx1)
{
  std::ostringstream ss;
  ss << prefix;
  if (append_epoch) ss << "e" << epoch << "_";
  ss << std::setw(digits) << std::setfill('0') << idx1;
  return ss.str();
}
} // namespace

class SpherePoseGeneratorNode : public rclcpp::Node
{
public:
  SpherePoseGeneratorNode()
  : rclcpp::Node("sphere_pose_generator_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // Frames
    base_frame_   = this->declare_parameter<std::string>("base_frame", "base_link");
    object_frame_ = this->declare_parameter<std::string>("object_frame", "object_frame");
    tcp_frame_    = this->declare_parameter<std::string>("tcp_frame", "tool0");
    camera_frame_ = this->declare_parameter<std::string>("camera_frame", "camera_link");

    // Object pose in base
    object_center_xyz_ = this->declare_parameter<std::vector<double>>("object_center_xyz", {0.5, 0.0, 0.0});
    object_orientation_quat_ = this->declare_parameter<std::vector<double>>(
      "object_orientation_quat", {0.0, 0.0, 0.0, 1.0});

    // Sphere sampling
    radius_m_ = this->declare_parameter<double>("radius_m", 0.30);
    n_poses_  = this->declare_parameter<int>("n_poses", 12);
    sampling_ = this->declare_parameter<std::string>("sampling", "fibonacci"); // fibonacci|latlong|random
    azimuth_deg_   = this->declare_parameter<std::vector<double>>("azimuth_deg", {0.0, 360.0});
    elevation_deg_ = this->declare_parameter<std::vector<double>>("elevation_deg", {25.0, 75.0});
    up_vector_base_ = this->declare_parameter<std::vector<double>>("up_vector_base", {0.0, 0.0, 1.0});
    seed_ = this->declare_parameter<int>("seed", 0);

    // Camera axes for look-at
    look_axis_cam_ = this->declare_parameter<std::vector<double>>("look_axis_cam", {0.0, 0.0, 1.0});
    up_axis_cam_   = this->declare_parameter<std::vector<double>>("up_axis_cam", {0.0, -1.0, 0.0});

    // TCP->Camera mount transform source
    use_tf_tcp_to_camera_ = this->declare_parameter<bool>("use_tf_tcp_to_camera", true);
    tcp_to_camera_xyz_  = this->declare_parameter<std::vector<double>>("tcp_to_camera_xyz", {0.0, 0.0, 0.0});
    tcp_to_camera_quat_ = this->declare_parameter<std::vector<double>>(
      "tcp_to_camera_quat", {0.0, 0.0, 0.0, 1.0});

    // Output naming / publishing
    pose_prefix_ = this->declare_parameter<std::string>("pose_prefix", "scan_pose_");
    pose_digits_ = this->declare_parameter<int>("pose_digits", 5);
    append_epoch_to_prefix_ = this->declare_parameter<bool>("append_epoch_to_prefix", true);
    publish_camera_frames_  = this->declare_parameter<bool>("publish_camera_frames", false);
    publish_tf_static_      = this->declare_parameter<bool>("publish_tf_static", true);
    auto_generate_on_startup_ = this->declare_parameter<bool>("auto_generate_on_startup", true);

    // Publishers
    poses_camera_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pose_generator/poses_camera", 1);
    poses_tcp_pub_    = this->create_publisher<geometry_msgs::msg::PoseArray>("/pose_generator/poses_tcp", 1);
    frame_names_pub_  = this->create_publisher<std_msgs::msg::String>("/pose_generator/frame_names", 1);

    // TF broadcaster
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    // Services
    generate_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/pose_generator/generate",
      std::bind(&SpherePoseGeneratorNode::on_generate, this, std::placeholders::_1, std::placeholders::_2));

    clear_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/pose_generator/clear",
      std::bind(&SpherePoseGeneratorNode::on_clear, this, std::placeholders::_1, std::placeholders::_2));

    if (auto_generate_on_startup_) {
      std::string err;
      if (!regenerate_and_publish(&err)) {
        RCLCPP_ERROR(this->get_logger(), "Auto-generate failed: %s", err.c_str());
      }
    }
  }

private:
  struct PoseItem {
    std::string tcp_name;
    std::string cam_name;
    tf2::Transform T_base_tcp;
    tf2::Transform T_base_cam;
  };

  void on_generate(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    // Refresh parameters (allow runtime updates)
    this->get_parameter("base_frame", base_frame_);
    this->get_parameter("object_frame", object_frame_);
    this->get_parameter("tcp_frame", tcp_frame_);
    this->get_parameter("camera_frame", camera_frame_);

    this->get_parameter("object_center_xyz", object_center_xyz_);
    this->get_parameter("object_orientation_quat", object_orientation_quat_);

    this->get_parameter("radius_m", radius_m_);
    this->get_parameter("n_poses", n_poses_);
    this->get_parameter("sampling", sampling_);
    this->get_parameter("azimuth_deg", azimuth_deg_);
    this->get_parameter("elevation_deg", elevation_deg_);
    this->get_parameter("up_vector_base", up_vector_base_);
    this->get_parameter("seed", seed_);

    this->get_parameter("look_axis_cam", look_axis_cam_);
    this->get_parameter("up_axis_cam", up_axis_cam_);

    this->get_parameter("use_tf_tcp_to_camera", use_tf_tcp_to_camera_);
    this->get_parameter("tcp_to_camera_xyz", tcp_to_camera_xyz_);
    this->get_parameter("tcp_to_camera_quat", tcp_to_camera_quat_);

    this->get_parameter("pose_prefix", pose_prefix_);
    this->get_parameter("pose_digits", pose_digits_);
    this->get_parameter("append_epoch_to_prefix", append_epoch_to_prefix_);
    this->get_parameter("publish_camera_frames", publish_camera_frames_);
    this->get_parameter("publish_tf_static", publish_tf_static_);

    std::string err;
    if (!regenerate_and_publish(&err)) {
      res->success = false;
      res->message = err;
      return;
    }

    res->success = true;
    std::ostringstream ss;
    ss << "Generated " << poses_.size() << " poses. epoch=" << epoch_;
    res->message = ss.str();
  }

  void on_clear(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    // NOTE: /tf_static is latched; old static transforms cannot be removed.
    // We handle "clear" by bumping epoch so new frame names are distinct.
    epoch_++;
    poses_.clear();

    res->success = true;
    res->message = "Cleared internal list (epoch incremented). Old /tf_static frames remain; use new frame names.";
  }

  bool regenerate_and_publish(std::string* err)
  {
    if (n_poses_ <= 0) {
      if (err) *err = "n_poses must be > 0";
      return false;
    }
    if (radius_m_ <= 0.0) {
      if (err) *err = "radius_m must be > 0";
      return false;
    }
    if (object_center_xyz_.size() != 3) {
      if (err) *err = "object_center_xyz must have size 3";
      return false;
    }
    if (azimuth_deg_.size() != 2 || elevation_deg_.size() != 2) {
      if (err) *err = "azimuth_deg and elevation_deg must have size 2";
      return false;
    }

    // Prepare transforms
    const tf2::Vector3 center_base(object_center_xyz_[0], object_center_xyz_[1], object_center_xyz_[2]);
    const tf2::Quaternion q_obj = quat_from_param(object_orientation_quat_, tf2::Quaternion(0,0,0,1));
    tf2::Transform T_base_object(q_obj, center_base);

    tf2::Transform T_tcp_cam;
    if (!get_tcp_to_camera_transform(&T_tcp_cam, err)) return false;

    // Sampling ranges
    double az0 = deg2rad(azimuth_deg_[0]);
    double az1 = deg2rad(azimuth_deg_[1]);
    double el0 = deg2rad(elevation_deg_[0]);
    double el1 = deg2rad(elevation_deg_[1]);

    if (az1 < az0) std::swap(az0, az1);
    if (el1 < el0) std::swap(el0, el1);

    // Clamp elevation to [-90, +90]
    el0 = std::clamp(el0, -M_PI_2, M_PI_2);
    el1 = std::clamp(el1, -M_PI_2, M_PI_2);

    const double az_range = std::max(1e-9, az1 - az0);

    // Camera axis parameters
    const tf2::Vector3 look_axis_cam = vec3_from_param(look_axis_cam_, tf2::Vector3(0,0,1));
    const tf2::Vector3 up_axis_cam   = vec3_from_param(up_axis_cam_,   tf2::Vector3(0,-1,0));
    const tf2::Vector3 up_base       = vec3_from_param(up_vector_base_, tf2::Vector3(0,0,1));

    // Generate directions in object frame (unit vectors center->camera)
    std::vector<tf2::Vector3> dirs_obj;
    dirs_obj.reserve(static_cast<size_t>(n_poses_));

    if (sampling_ == "random") {
      std::mt19937 rng(seed_ == 0 ? std::random_device{}() : static_cast<uint32_t>(seed_));
      std::uniform_real_distribution<double> u01(0.0, 1.0);
      std::uniform_real_distribution<double> uaz(az0, az1);

      const double zmin = std::sin(el0);
      const double zmax = std::sin(el1);

      for (int i = 0; i < n_poses_; ++i) {
        const double z = zmin + (zmax - zmin) * u01(rng);
        const double r = std::sqrt(std::max(0.0, 1.0 - z*z));
        const double phi = uaz(rng);

        dirs_obj.emplace_back(r * std::cos(phi), r * std::sin(phi), z);
      }
    } else if (sampling_ == "latlong") {
      const int n_el = std::max(1, static_cast<int>(std::round(std::sqrt(static_cast<double>(n_poses_)))));
      const int n_az = std::max(1, static_cast<int>(std::ceil(static_cast<double>(n_poses_) / n_el)));

      int count = 0;
      for (int j = 0; j < n_el && count < n_poses_; ++j) {
        const double t_el = (j + 0.5) / static_cast<double>(n_el);
        const double el = el0 + (el1 - el0) * t_el;
        const double z = std::sin(el);
        const double r = std::cos(el);

        for (int i = 0; i < n_az && count < n_poses_; ++i) {
          const double t_az = (i + 0.5) / static_cast<double>(n_az);
          const double phi = az0 + az_range * t_az;

          dirs_obj.emplace_back(r * std::cos(phi), r * std::sin(phi), z);
          count++;
        }
      }
    } else { // default: fibonacci
      // Low-discrepancy on azimuth + linear in z on [sin(el0), sin(el1)]
      const double zmin = std::sin(el0);
      const double zmax = std::sin(el1);
      const double gr_conj = (std::sqrt(5.0) - 1.0) / 2.0; // ~0.618

      for (int i = 0; i < n_poses_; ++i) {
        const double t = (i + 0.5) / static_cast<double>(n_poses_);
        const double z = zmin + (zmax - zmin) * t;
        const double r = std::sqrt(std::max(0.0, 1.0 - z*z));

        const double frac = std::fmod(i * gr_conj, 1.0);
        const double phi = az0 + az_range * frac;

        dirs_obj.emplace_back(r * std::cos(phi), r * std::sin(phi), z);
      }
    }

    // Build pose list
    std::vector<PoseItem> poses;
    poses.reserve(dirs_obj.size());

    for (int i = 0; i < static_cast<int>(dirs_obj.size()); ++i) {
      const tf2::Vector3 dir_obj = dirs_obj[static_cast<size_t>(i)];
      tf2::Vector3 p_obj = dir_obj * radius_m_; // camera position around object center, in object frame

      // Convert to base
      const tf2::Vector3 p_base = T_base_object * p_obj;

      // Forward vector from camera to center
      tf2::Vector3 f_base = center_base - p_base;
      if (f_base.length2() < kEps) continue;
      f_base.normalize();

      const tf2::Quaternion q_base_cam = compute_lookat_quat(f_base, up_base, look_axis_cam, up_axis_cam);
      tf2::Transform T_base_cam(q_base_cam, p_base);

      // Desired TCP pose: T_base_tcp * T_tcp_cam = T_base_cam  => T_base_tcp = T_base_cam * inv(T_tcp_cam)
      tf2::Transform T_base_tcp = T_base_cam * T_tcp_cam.inverse();

      PoseItem item;
      item.tcp_name = make_name(pose_prefix_, append_epoch_to_prefix_, epoch_, pose_digits_, i + 1);
      item.cam_name = item.tcp_name + "_cam";
      item.T_base_tcp = T_base_tcp;
      item.T_base_cam = T_base_cam;
      poses.push_back(item);
    }

    if (poses.empty()) {
      if (err) *err = "No poses generated (check ranges / radius / parameters).";
      return false;
    }

    poses_ = std::move(poses);

    // Publish TF + PoseArray + names
    publish_object_frame_tf(T_base_object);
    publish_pose_frames_tf();
    publish_pose_arrays();
    publish_frame_names();

    return true;
  }

  bool get_tcp_to_camera_transform(tf2::Transform* out, std::string* err)
  {
    if (!out) return false;

    if (use_tf_tcp_to_camera_) {
      try {
        // lookupTransform(target, source): gives T_target_source
        // We need T_tcp_camera, so target=tcp_frame, source=camera_frame
        const auto tf = tf_buffer_.lookupTransform(
          tcp_frame_, camera_frame_, tf2::TimePointZero,
          tf2::durationFromSec(0.2));

        tf2::Transform T;
        tf2::fromMsg(tf.transform, T);
        *out = T;
        return true;
      } catch (const std::exception& e) {
        if (err) {
          std::ostringstream ss;
          ss << "TF lookup failed for T_" << tcp_frame_ << "_" << camera_frame_
             << ": " << e.what()
             << ". Provide tcp_to_camera_xyz/quaternion or set use_tf_tcp_to_camera=false.";
          *err = ss.str();
        }
        return false;
      }
    }

    // From parameters
    const tf2::Vector3 t = vec3_from_param(tcp_to_camera_xyz_, tf2::Vector3(0,0,0));
    const tf2::Quaternion q = quat_from_param(tcp_to_camera_quat_, tf2::Quaternion(0,0,0,1));
    *out = tf2::Transform(q, t);
    return true;
  }

  void publish_object_frame_tf(const tf2::Transform& T_base_object)
  {
    geometry_msgs::msg::TransformStamped ts;
    ts.header.stamp = this->now();
    ts.header.frame_id = base_frame_;
    ts.child_frame_id = object_frame_;
    ts.transform = tf2::toMsg(T_base_object);

    if (publish_tf_static_) {
      static_broadcaster_->sendTransform(ts);
    }
  }

  void publish_pose_frames_tf()
  {
    if (!publish_tf_static_) return;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    tfs.reserve(poses_.size() * (publish_camera_frames_ ? 2 : 1));

    const auto stamp = this->now();

    for (const auto& p : poses_) {
      geometry_msgs::msg::TransformStamped ts_tcp;
      ts_tcp.header.stamp = stamp;
      ts_tcp.header.frame_id = base_frame_;
      ts_tcp.child_frame_id = p.tcp_name;
      ts_tcp.transform = tf2::toMsg(p.T_base_tcp);
      tfs.push_back(ts_tcp);

      if (publish_camera_frames_) {
        geometry_msgs::msg::TransformStamped ts_cam;
        ts_cam.header.stamp = stamp;
        ts_cam.header.frame_id = base_frame_;
        ts_cam.child_frame_id = p.cam_name;
        ts_cam.transform = tf2::toMsg(p.T_base_cam);
        tfs.push_back(ts_cam);
      }
    }

    static_broadcaster_->sendTransform(tfs);
  }

  void publish_pose_arrays()
  {
    geometry_msgs::msg::PoseArray pa_cam;
    geometry_msgs::msg::PoseArray pa_tcp;
    pa_cam.header.stamp = this->now();
    pa_cam.header.frame_id = base_frame_;
    pa_tcp.header = pa_cam.header;

    pa_cam.poses.reserve(poses_.size());
    pa_tcp.poses.reserve(poses_.size());

    for (const auto& p : poses_) {
      geometry_msgs::msg::Pose pose_cam;
      pose_cam.position.x = p.T_base_cam.getOrigin().x();
      pose_cam.position.y = p.T_base_cam.getOrigin().y();
      pose_cam.position.z = p.T_base_cam.getOrigin().z();
      pose_cam.orientation = tf2::toMsg(p.T_base_cam.getRotation());
      pa_cam.poses.push_back(pose_cam);

      geometry_msgs::msg::Pose pose_tcp;
      pose_tcp.position.x = p.T_base_tcp.getOrigin().x();
      pose_tcp.position.y = p.T_base_tcp.getOrigin().y();
      pose_tcp.position.z = p.T_base_tcp.getOrigin().z();
      pose_tcp.orientation = tf2::toMsg(p.T_base_tcp.getRotation());
      pa_tcp.poses.push_back(pose_tcp);
    }

    poses_camera_pub_->publish(pa_cam);
    poses_tcp_pub_->publish(pa_tcp);
  }

  void publish_frame_names()
  {
    std::vector<std::string> names;
    names.reserve(poses_.size());
    for (const auto& p : poses_) names.push_back(p.tcp_name);

    std_msgs::msg::String msg;
    msg.data = json_string_list(names);
    frame_names_pub_->publish(msg);
  }

private:
  // Params (cached)
  std::string base_frame_;
  std::string object_frame_;
  std::string tcp_frame_;
  std::string camera_frame_;

  std::vector<double> object_center_xyz_;
  std::vector<double> object_orientation_quat_;

  double radius_m_;
  int n_poses_;
  std::string sampling_;
  std::vector<double> azimuth_deg_;
  std::vector<double> elevation_deg_;
  std::vector<double> up_vector_base_;
  int seed_;

  std::vector<double> look_axis_cam_;
  std::vector<double> up_axis_cam_;

  bool use_tf_tcp_to_camera_;
  std::vector<double> tcp_to_camera_xyz_;
  std::vector<double> tcp_to_camera_quat_;

  std::string pose_prefix_;
  int pose_digits_;
  bool append_epoch_to_prefix_;
  bool publish_camera_frames_;
  bool publish_tf_static_;
  bool auto_generate_on_startup_;

  int epoch_{0};

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;

  // Data
  std::vector<PoseItem> poses_;

  // ROS
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr poses_camera_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr poses_tcp_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr frame_names_pub_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr generate_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SpherePoseGeneratorNode>());
  rclcpp::shutdown();
  return 0;
}
