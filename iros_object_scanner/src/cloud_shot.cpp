// iros_object_scanner/src/cloud_snapshot_node.cpp
//
// ROS2 Humble: snapshot + filtering for Azure Kinect point cloud (KEEP COLORS)
// Input : /point2 (frame rgb_camera_link, must contain rgb/rgba fields to preserve colors)
// Output: /scanner/object_cloud (frame base)
// Service: /scanner/capture (std_srvs/Trigger)
//
// Pipeline: TF -> ROI CropBox -> (optional VoxelGrid w/o averaging packed RGB)
//           -> RANSAC plane remove -> Z-cut -> largest cluster -> outlier removal -> publish
//
// Default: async_capture=true to avoid service response timeout.
// MultiThreadedExecutor is used in main().

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "rmw/qos_profiles.h"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/filter.h>  // removeNaNFromPointCloud
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/search/kdtree.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

class CloudSnapshotNode : public rclcpp::Node
{
public:
  using PointT = pcl::PointXYZRGB;
  using CloudT = pcl::PointCloud<PointT>;

  CloudSnapshotNode()
  : Node("cloud_snapshot_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    cbg_sub_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    cbg_srv_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // --- params ---
    input_cloud_topic_  = this->declare_parameter<std::string>("input_cloud_topic", "/point2");
    target_frame_       = this->declare_parameter<std::string>("target_frame", "base");
    output_cloud_topic_ = this->declare_parameter<std::string>("output_cloud_topic", "/scanner/object_cloud");

    publish_debug_      = this->declare_parameter<bool>("publish_debug", true);
    async_capture_      = this->declare_parameter<bool>("async_capture", true);

    max_cloud_age_ms_   = this->declare_parameter<int>("max_cloud_age_ms", 200);
    tf_timeout_ms_      = this->declare_parameter<int>("tf_timeout_ms", 200);

    // ROI in target_frame (meters)
    roi_x_min_ = this->declare_parameter<double>("roi_x_min", -0.20);
    roi_x_max_ = this->declare_parameter<double>("roi_x_max",  0.20);
    roi_y_min_ = this->declare_parameter<double>("roi_y_min", -0.20);
    roi_y_max_ = this->declare_parameter<double>("roi_y_max",  0.20);
    roi_z_min_ = this->declare_parameter<double>("roi_z_min",  0.00);
    roi_z_max_ = this->declare_parameter<double>("roi_z_max",  0.35);

    // Voxel: keep colors (do NOT average packed RGB). We keep representative color.
    use_voxel_    = this->declare_parameter<bool>("use_voxel", true);
    voxel_leaf_m_ = this->declare_parameter<double>("voxel_leaf_m", 0.002);

    // Plane (table)
    ransac_dist_thresh_ = this->declare_parameter<double>("ransac_dist_thresh", 0.005);
    ransac_max_iter_    = this->declare_parameter<int>("ransac_max_iter", 1000);
    table_margin_z_     = this->declare_parameter<double>("table_margin_z", 0.005);
    min_plane_inliers_  = this->declare_parameter<int>("min_plane_inliers", 3000);

    // Clustering
    cluster_tolerance_  = this->declare_parameter<double>("cluster_tolerance", 0.010);
    min_cluster_size_   = this->declare_parameter<int>("min_cluster_size", 800);
    max_cluster_size_   = this->declare_parameter<int>("max_cluster_size", 500000);

    // Outliers
    outlier_method_     = this->declare_parameter<std::string>("outlier_method", "sor"); // sor|radius|none
    sor_mean_k_         = this->declare_parameter<int>("sor_mean_k", 50);
    sor_stddev_mul_     = this->declare_parameter<double>("sor_stddev_mul", 1.0);
    radius_outlier_r_   = this->declare_parameter<double>("radius_outlier_r", 0.005);
    radius_outlier_min_ = this->declare_parameter<int>("radius_outlier_min_neighbors", 5);

    // --- subscription ---
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = cbg_sub_;

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_cloud_topic_, rclcpp::SensorDataQoS(),
      std::bind(&CloudSnapshotNode::on_cloud, this, std::placeholders::_1),
      sub_opts);

    // --- publishers ---
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_cloud_topic_, 10);

    if (publish_debug_) {
      pub_debug_roi_      = this->create_publisher<sensor_msgs::msg::PointCloud2>("/scanner/debug/roi", 10);
      pub_debug_no_table_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/scanner/debug/no_table", 10);
      pub_debug_table_    = this->create_publisher<sensor_msgs::msg::PointCloud2>("/scanner/debug/table_plane", 10);
    }

    // --- service ---
    capture_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/scanner/capture",
      std::bind(&CloudSnapshotNode::on_capture, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default,
      cbg_srv_);

    RCLCPP_INFO(get_logger(),
      "CloudSnapshotNode ready. input=%s -> %s (frame=%s), async_capture=%s",
      input_cloud_topic_.c_str(), output_cloud_topic_.c_str(), target_frame_.c_str(),
      async_capture_ ? "true" : "false");
  }

private:
  // ---------- ROS callbacks ----------
  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(last_mutex_);
    last_cloud_ = msg;
  }

  void on_capture(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                  std::shared_ptr<std_srvs::srv::Trigger::Response> resp)
  {
    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) {
      resp->success = false;
      resp->message = "Busy: previous capture still processing.";
      return;
    }

    auto cloud_msg = get_latest_cloud_checked(resp);
    if (!cloud_msg) {
      busy_ = false;
      return;
    }

    if (async_capture_) {
      resp->success = true;
      resp->message = "Accepted. Processing in background; result will be published.";

      std::thread([this, cloud_msg]() {
        try {
          process_and_publish(*cloud_msg);
        } catch (const std::exception &e) {
          RCLCPP_ERROR(this->get_logger(), "Processing failed: %s", e.what());
        }
        busy_ = false;
      }).detach();

      return;
    }

    try {
      process_and_publish(*cloud_msg);
      resp->success = true;
      resp->message = "OK. Published.";
    } catch (const std::exception &e) {
      resp->success = false;
      resp->message = std::string("Failed: ") + e.what();
    }
    busy_ = false;
  }

  // ---------- helpers ----------
  std::optional<sensor_msgs::msg::PointCloud2> get_latest_cloud_checked(
    const std::shared_ptr<std_srvs::srv::Trigger::Response> &resp)
  {
    sensor_msgs::msg::PointCloud2::SharedPtr last;
    {
      std::lock_guard<std::mutex> lk(last_mutex_);
      last = last_cloud_;
    }

    if (!last) {
      resp->success = false;
      resp->message = "No PointCloud2 received yet.";
      return std::nullopt;
    }

    // freshness check (if stamp != 0)
    if (max_cloud_age_ms_ > 0) {
      if (last->header.stamp.sec != 0 || last->header.stamp.nanosec != 0) {
        rclcpp::Time t_cloud(last->header.stamp);
        rclcpp::Time t_now = this->now();
        auto age_ms = (t_now - t_cloud).nanoseconds() / 1000000LL;
        if (age_ms > max_cloud_age_ms_) {
          resp->success = false;
          resp->message = "Cloud too old: age_ms=" + std::to_string((long long)age_ms) +
                          " > " + std::to_string(max_cloud_age_ms_);
          return std::nullopt;
        }
      }
    }

    // copy (safe for async processing)
    sensor_msgs::msg::PointCloud2 copy = *last;
    return copy;
  }

  void transform_cloud_to_target(const sensor_msgs::msg::PointCloud2 &cloud_in,
                                 sensor_msgs::msg::PointCloud2 &cloud_out)
  {
    auto timeout = rclcpp::Duration::from_nanoseconds((int64_t)tf_timeout_ms_ * 1000000LL);

    const bool has_stamp =
      (cloud_in.header.stamp.sec != 0 || cloud_in.header.stamp.nanosec != 0);

    rclcpp::Time stamp = has_stamp
      ? rclcpp::Time(cloud_in.header.stamp)
      : rclcpp::Time(0, 0, this->get_clock()->get_clock_type());

    try {
      if (has_stamp) {
        if (!tf_buffer_.canTransform(target_frame_, cloud_in.header.frame_id, stamp, timeout)) {
          stamp = rclcpp::Time(0, 0, this->get_clock()->get_clock_type()); // latest
        }
      }

      auto tf = tf_buffer_.lookupTransform(
        target_frame_, cloud_in.header.frame_id, stamp, timeout);

      tf2::doTransform(cloud_in, cloud_out, tf);
      cloud_out.header.frame_id = target_frame_;
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("TF transform failed: ") + e.what());
    }
  }

  void publish_debug_cloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub,
                           const CloudT &cloud,
                           const std_msgs::msg::Header &hdr)
  {
    if (!publish_debug_ || !pub) return;
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header = hdr;
    pub->publish(msg);
  }

  void process_and_publish(const sensor_msgs::msg::PointCloud2 &cloud_in)
  {
    // 1) TF -> target frame
    sensor_msgs::msg::PointCloud2 cloud_tf;
    transform_cloud_to_target(cloud_in, cloud_tf);

    // 2) ROS -> PCL (XYZRGB)
    CloudT::Ptr cloud(new CloudT());
    pcl::fromROSMsg(cloud_tf, *cloud);
    if (cloud->empty()) throw std::runtime_error("Input cloud empty after TF.");

    // Remove NaN
    {
      std::vector<int> idx;
      pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);
    }
    if (cloud->empty()) throw std::runtime_error("All points are NaN.");

    // 3) ROI CropBox
    CloudT::Ptr cloud_roi(new CloudT());
    {
      pcl::CropBox<PointT> crop;
      crop.setInputCloud(cloud);
      crop.setMin(Eigen::Vector4f((float)roi_x_min_, (float)roi_y_min_, (float)roi_z_min_, 1.0f));
      crop.setMax(Eigen::Vector4f((float)roi_x_max_, (float)roi_y_max_, (float)roi_z_max_, 1.0f));
      crop.filter(*cloud_roi);
    }
    if (cloud_roi->empty()) throw std::runtime_error("ROI crop empty. Tune roi_*.");

    publish_debug_cloud(pub_debug_roi_, *cloud_roi, cloud_tf.header);

    // 4) Voxel (optional) - DO NOT average packed RGB
    CloudT::Ptr cloud_ds(new CloudT());
    if (use_voxel_ && voxel_leaf_m_ > 0.0) {
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(cloud_roi);
      vg.setLeafSize((float)voxel_leaf_m_, (float)voxel_leaf_m_, (float)voxel_leaf_m_);
      vg.setDownsampleAllData(false);  // critical: keep representative color, do not average packed rgb
      vg.filter(*cloud_ds);
    } else {
      cloud_ds = cloud_roi;
    }
    if (cloud_ds->empty()) throw std::runtime_error("Empty after voxel.");

    // 5) Plane segmentation (table) via RANSAC
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());
    bool plane_found = false;

    {
      pcl::SACSegmentation<PointT> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setMaxIterations(ransac_max_iter_);
      seg.setDistanceThreshold(ransac_dist_thresh_);
      seg.setInputCloud(cloud_ds);
      seg.segment(*inliers, *coeff);

      if (!inliers->indices.empty() && (int)inliers->indices.size() >= min_plane_inliers_) {
        plane_found = true;
      }
    }

    CloudT::Ptr cloud_no_table(new CloudT());
    CloudT::Ptr cloud_table(new CloudT());
    double table_z_mean = std::numeric_limits<double>::quiet_NaN();

    if (plane_found) {
      pcl::ExtractIndices<PointT> ex;
      ex.setInputCloud(cloud_ds);
      ex.setIndices(inliers);

      ex.setNegative(false);
      ex.filter(*cloud_table);

      ex.setNegative(true);
      ex.filter(*cloud_no_table);

      if (!cloud_table->empty()) {
        double sum_z = 0.0;
        for (const auto &p : cloud_table->points) sum_z += p.z;
        table_z_mean = sum_z / (double)cloud_table->points.size();
      }
    } else {
      cloud_no_table = cloud_ds; // fallback
    }

    publish_debug_cloud(pub_debug_table_, *cloud_table, cloud_tf.header);

    if (cloud_no_table->empty()) throw std::runtime_error("Empty after removing plane.");

    // 6) Z-cut above table (remove rims/leftovers)
    CloudT::Ptr cloud_zcut(new CloudT());
    {
      pcl::PassThrough<PointT> pass;
      pass.setInputCloud(cloud_no_table);
      pass.setFilterFieldName("z");

      if (plane_found && std::isfinite(table_z_mean)) {
        pass.setFilterLimits((float)(table_z_mean + table_margin_z_), (float)roi_z_max_);
      } else {
        pass.setFilterLimits((float)roi_z_min_, (float)roi_z_max_);
      }
      pass.filter(*cloud_zcut);
    }

    publish_debug_cloud(pub_debug_no_table_, *cloud_zcut, cloud_tf.header);

    if (cloud_zcut->empty()) throw std::runtime_error("Empty after z-cut. Tune table_margin_z/ROI.");

    // 7) Largest cluster (object)
    CloudT::Ptr cloud_obj(new CloudT());
    {
      pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
      tree->setInputCloud(cloud_zcut);

      std::vector<pcl::PointIndices> clusters;
      pcl::EuclideanClusterExtraction<PointT> ec;
      ec.setClusterTolerance(cluster_tolerance_);
      ec.setMinClusterSize(min_cluster_size_);
      ec.setMaxClusterSize(max_cluster_size_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(cloud_zcut);
      ec.extract(clusters);

      if (clusters.empty()) {
        cloud_obj = cloud_zcut; // fallback
      } else {
        size_t best_i = 0, best_sz = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
          if (clusters[i].indices.size() > best_sz) {
            best_sz = clusters[i].indices.size();
            best_i = i;
          }
        }

        pcl::ExtractIndices<PointT> ex;
        ex.setInputCloud(cloud_zcut);
        pcl::PointIndices::Ptr idx(new pcl::PointIndices(clusters[best_i]));
        ex.setIndices(idx);
        ex.setNegative(false);
        ex.filter(*cloud_obj);
      }
    }
    if (cloud_obj->empty()) throw std::runtime_error("Empty after clustering. Tune cluster_*.");

    // 8) Outlier removal
    CloudT::Ptr cloud_final(new CloudT());
    if (outlier_method_ == "none") {
      cloud_final = cloud_obj;
    } else if (outlier_method_ == "radius") {
      pcl::RadiusOutlierRemoval<PointT> ror;
      ror.setInputCloud(cloud_obj);
      ror.setRadiusSearch(radius_outlier_r_);
      ror.setMinNeighborsInRadius(radius_outlier_min_);
      ror.filter(*cloud_final);
    } else { // default sor
      pcl::StatisticalOutlierRemoval<PointT> sor;
      sor.setInputCloud(cloud_obj);
      sor.setMeanK(sor_mean_k_);
      sor.setStddevMulThresh(sor_stddev_mul_);
      sor.filter(*cloud_final);
    }
    if (cloud_final->empty()) throw std::runtime_error("Empty after outlier removal.");

    // 9) Publish result (keep colors)
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*cloud_final, out);
    out.header = cloud_tf.header; // frame=target_frame
    cloud_pub_->publish(out);

    RCLCPP_INFO(get_logger(), "Published colored object cloud: points=%zu plane=%d",
                cloud_final->size(), plane_found ? 1 : 0);
  }

private:
  rclcpp::CallbackGroup::SharedPtr cbg_sub_;
  rclcpp::CallbackGroup::SharedPtr cbg_srv_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_roi_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_no_table_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_table_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr capture_srv_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex last_mutex_;
  sensor_msgs::msg::PointCloud2::SharedPtr last_cloud_{nullptr};

  std::atomic<bool> busy_{false};

  std::string input_cloud_topic_;
  std::string target_frame_;
  std::string output_cloud_topic_;

  bool publish_debug_{true};
  bool async_capture_{true};

  int max_cloud_age_ms_{200};
  int tf_timeout_ms_{200};

  double roi_x_min_, roi_x_max_;
  double roi_y_min_, roi_y_max_;
  double roi_z_min_, roi_z_max_;

  bool use_voxel_{true};
  double voxel_leaf_m_{0.002};

  double ransac_dist_thresh_{0.005};
  int ransac_max_iter_{1000};
  double table_margin_z_{0.005};
  int min_plane_inliers_{3000};

  double cluster_tolerance_{0.010};
  int min_cluster_size_{800};
  int max_cluster_size_{500000};

  std::string outlier_method_{"sor"};
  int sor_mean_k_{50};
  double sor_stddev_mul_{1.0};
  double radius_outlier_r_{0.005};
  int radius_outlier_min_{5};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CloudSnapshotNode>();

  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
