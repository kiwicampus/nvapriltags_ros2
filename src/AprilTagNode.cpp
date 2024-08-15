/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cv_bridge/cv_bridge.h>

#include <AprilTagNode.hpp>
#include <Eigen/Dense>
#include <image_transport/image_transport.hpp>

#include "cuda.h"          // NOLINT
#include "cuda_runtime.h"  // NOLINT
#include "nvAprilTags.h"

namespace {
geometry_msgs::msg::Transform ToTransformMsg(const nvAprilTagsID_t &detection) {
  geometry_msgs::msg::Transform t;
  t.translation.x = detection.translation[0];
  t.translation.y = detection.translation[1];
  t.translation.z = detection.translation[2];

  // Rotation matrix from nvAprilTags is column major
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::ColMajor>>
      orientation(detection.orientation);
  const Eigen::Quaternion<float> q(orientation);

  t.rotation.w = q.w();
  t.rotation.x = q.x();
  t.rotation.y = q.y();
  t.rotation.z = q.z();

  return t;
}
}  // namespace

struct AprilTagNode::AprilTagsImpl {
  // Handle used to interface with the stereo library.
  nvAprilTagsHandle april_tags_handle = nullptr;

  // Camera intrinsics
  nvAprilTagsCameraIntrinsics_t cam_intrinsics;

  // Output vector of detected Tags
  std::vector<nvAprilTagsID_t> tags;

  // CUDA stream
  cudaStream_t main_stream = {};

  // CUDA buffers to store the input image.
  nvAprilTagsImageInput_t input_image;

  // CUDA memory buffer container for RGBA images.
  char *input_image_buffer = nullptr;

  // Size of image buffer
  size_t input_image_buffer_size = 0;

  void initialize(const AprilTagNode &node, const uint32_t width,
                  const uint32_t height, const size_t image_buffer_size,
                  const size_t pitch_bytes,
                  const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg_ci) {
    assert(april_tags_handle == nullptr && "Already initialized.");

    // Get camera intrinsics
    const double *k = msg_ci->k.data();
    const float fx = static_cast<float>(k[0]);
    const float fy = static_cast<float>(k[4]);
    const float cx = static_cast<float>(k[2]);
    const float cy = static_cast<float>(k[5]);
    cam_intrinsics = {fx, fy, cx, cy};

    // Create AprilTags detector instance and get handle
    const int error = nvCreateAprilTagsDetector(
        &april_tags_handle, width, height, nvAprilTagsFamily::NVAT_TAG36H11,
        &cam_intrinsics, node.tag_edge_size_);
    if (error != 0) {
      throw std::runtime_error(
          "Failed to create NV April Tags detector (error code " +
          std::to_string(error) + ")");
    }

    // Create stream for detection
    cudaStreamCreate(&main_stream);

    // Allocate the output vector to contain detected AprilTags.
    tags.resize(node.max_tags_);

    // Setup input image CUDA buffer.
    const cudaError_t cuda_error =
        cudaMalloc(&input_image_buffer, image_buffer_size);
    if (cuda_error != cudaSuccess) {
      throw std::runtime_error("Could not allocate CUDA memory (error code " +
                               std::to_string(cuda_error) + ")");
    }

    // Setup input image.
    input_image_buffer_size = image_buffer_size;
    input_image.width = width;
    input_image.height = height;
    input_image.dev_ptr = reinterpret_cast<uchar4 *>(input_image_buffer);
    input_image.pitch = pitch_bytes;
  }

  ~AprilTagsImpl() {
    if (april_tags_handle != nullptr) {
      cudaStreamDestroy(main_stream);
      nvAprilTagsDestroy(april_tags_handle);
      cudaFree(input_image_buffer);
    }
  }
};

AprilTagNode::AprilTagNode(rclcpp::NodeOptions options)
    : Node("apriltag", "apriltag", options.use_intra_process_comms(true)),
      // parameter
      // tag_family_(declare_parameter<std::string>("family", "36h11")),
      // tag_edge_size_(declare_parameter<double>("size", 2.0)),
      // max_tags_(declare_parameter<int>("max_tags", 20)),
      // topics
      // sub_cam_(image_transport::create_camera_subscription(
      //     this, "image",
      //     std::bind(&AprilTagNode::onCameraFrame, this, std::placeholders::_1,
      //               std::placeholders::_2),
      //     declare_parameter<std::string>("image_transport", "raw"),
      //     rmw_qos_profile_sensor_data)),
      pub_tf_(
          create_publisher<tf2_msgs::msg::TFMessage>("/tf", rclcpp::QoS(100))),
      pub_detections_(
          create_publisher<nvapriltags_ros2::msg::AprilTagDetectionArray>(
              "detections", rclcpp::QoS(1))),
      impl_(std::make_unique<AprilTagsImpl>()) 
      {
        param_manager_ = NodeParamManager(this);
        param_manager_.addParameter<std::string>(tag_family_, "family", "36h11");
        param_manager_.addParameter<double>(tag_edge_size_, "size", 2.0);
        param_manager_.addParameter<int>(max_tags_, "max_tags", 20);
        param_manager_.addParameter<int>(throttle_interval_ms_, "processing_period_ms", 100);
        params_callback_handle_ = this->add_on_set_parameters_callback(
          std::bind(&AprilTagNode::parameters_cb, this, std::placeholders::_1));

        detection_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false);
        // Add camera info and image subscriptions to the callback group
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = detection_callback_group_;
        // Create separate subscriptions for camera info and image
        sub_cam_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "camera_info", rclcpp::QoS(1).best_effort(), std::bind(&AprilTagNode::onCameraInfo, this, std::placeholders::_1), sub_options);

        sub_cam_ = create_subscription<sensor_msgs::msg::Image>(
            "image", rclcpp::QoS(1).best_effort(), std::bind(&AprilTagNode::onCameraFrame, this, std::placeholders::_1), sub_options);

        detection_exec_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        detection_exec_->add_callback_group(detection_callback_group_, get_node_base_interface());

        processing_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(throttle_interval_ms_),
            std::bind(&AprilTagNode::processImages, this));

        // Service to start/stop processing
        srv_enable_processing_ = create_service<std_srvs::srv::SetBool>(
            "control_processing",
            std::bind(&AprilTagNode::controlProcessing, this, std::placeholders::_1, std::placeholders::_2));
      }

void AprilTagNode::processImages() {
  if (enable_processing_) {
    detection_exec_->spin_some();
  }
}

void AprilTagNode::onCameraFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_img) {
  cv::Mat img_rgb8 = cv_bridge::toCvShare(msg_img, "rgb8")->image;

  // Create an empty RGBA image with the same size as the input image
  cv::Mat img_rgba8;

  // Convert the RGB image to RGBA by adding an alpha channel
  cv::cvtColor(img_rgb8, img_rgba8, cv::COLOR_RGB2RGBA);

  if(saved_cam_info_)
  {
  // Setup detector on first frame
    if (impl_->april_tags_handle == nullptr) {
      impl_->initialize(*this, img_rgba8.cols, img_rgba8.rows,
                        img_rgba8.total() * img_rgba8.elemSize(), img_rgba8.step,
                        saved_cam_info_);
      RCLCPP_INFO(get_logger(), "Apriltag detection initialized on %s, unsubscribing from %s", sub_cam_->get_topic_name(), sub_cam_info_->get_topic_name());
      // After initialization, stop subscribing to camera info
      saved_cam_info_.reset();
      system_initialized_ = true;
    }
  }
  else
  {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 20000, "Waiting for camera info on topic %s", sub_cam_->get_topic_name());
  }

  // Copy frame into CUDA buffer
  const cudaError_t cuda_error =
      cudaMemcpy(impl_->input_image_buffer, img_rgba8.ptr(),
                 impl_->input_image_buffer_size, cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) {
    throw std::runtime_error(
        "Could not memcpy to device CUDA memory (error code " +
        std::to_string(cuda_error) + ")");
  }

  // Perform detection
  uint32_t num_detections;
  const int error = nvAprilTagsDetect(
      impl_->april_tags_handle, &(impl_->input_image), impl_->tags.data(),
      &num_detections, max_tags_, impl_->main_stream);
  if (error != 0) {
    throw std::runtime_error("Failed to run AprilTags detector (error code " +
                             std::to_string(error) + ")");
  }

  // Parse detections into published protos
  nvapriltags_ros2::msg::AprilTagDetectionArray msg_detections;
  msg_detections.header = msg_img->header;
  tf2_msgs::msg::TFMessage tfs;
  for (int i = 0; i < num_detections; i++) {
    const nvAprilTagsID_t &detection = impl_->tags[i];

    // detection
    nvapriltags_ros2::msg::AprilTagDetection msg_detection;
    msg_detection.family = tag_family_;
    msg_detection.id = detection.id;

    // corners
    for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
      msg_detection.corners.data()[corner_idx].x =
          detection.corners[corner_idx].x;
      msg_detection.corners.data()[corner_idx].y =
          detection.corners[corner_idx].y;
    }

    // center
    const float slope_1 = (detection.corners[2].y - detection.corners[0].y) /
                    (detection.corners[2].x - detection.corners[0].x);
    const float slope_2 = (detection.corners[3].y - detection.corners[1].y) /
                      (detection.corners[3].x - detection.corners[1].x);
    const float intercept_1 = detection.corners[0].y -
                              (slope_1 * detection.corners[0].x);
    const float intercept_2 = detection.corners[3].y -
                              (slope_2 * detection.corners[3].x);
    msg_detection.center.x = (intercept_2 - intercept_1) / (slope_1 - slope_2);
    msg_detection.center.y = (slope_2 * intercept_1 - slope_1 * intercept_2) /
                              (slope_2 - slope_1);

    // Timestamped Pose3 transform
    geometry_msgs::msg::TransformStamped tf;
    tf.header = msg_img->header;
    tf.child_frame_id =
        std::string(tag_family_) + ":" + std::to_string(detection.id);
    tf.transform = ToTransformMsg(detection);
    tfs.transforms.push_back(tf);

    // Pose
    msg_detection.pose.pose.pose.position.x = tf.transform.translation.x;
    msg_detection.pose.pose.pose.position.y = tf.transform.translation.y;
    msg_detection.pose.pose.pose.position.z = tf.transform.translation.z;
    msg_detection.pose.pose.pose.orientation = tf.transform.rotation;
    msg_detections.detections.push_back(msg_detection);
  }

  if(pub_detections_->get_subscription_count())
  {
    pub_detections_->publish(msg_detections);
  }
  pub_tf_->publish(tfs);
}

void AprilTagNode::onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg_ci) {
  saved_cam_info_ = msg_ci;
}

void AprilTagNode::controlProcessing(const std_srvs::srv::SetBool::Request::SharedPtr request,
                        std_srvs::srv::SetBool::Response::SharedPtr response) {
  if(!system_initialized_)
  {
    response->success = false;
    RCLCPP_ERROR(this->get_logger(), "Apriltag detector on %s cannot be paused because its not initalized", sub_cam_->get_topic_name());
    response->message = "Execution cannot be paused if the system has not initialized!";
  }                          
  enable_processing_ = request->data;
  response->success = true;
  response->message = enable_processing_ ? "Processing enabled" : "Processing disabled";
}

rcl_interfaces::msg::SetParametersResult AprilTagNode::parameters_cb(const std::vector<rclcpp::Parameter>& parameters)
{
  // Prevent updating parameters while processing
  auto result = param_manager_.parametersCb(parameters);
  for (auto& parameter : parameters)
  {
    const auto& type = parameter.get_type();
    const auto& name = parameter.get_name();

    if (type == rclcpp::ParameterType::PARAMETER_INTEGER)
    {
      if (name == "processing_period_ms")
      {
        throttle_interval_ms_ = parameter.as_int();
          processing_timer_->reset();
          processing_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(throttle_interval_ms_),
            std::bind(&AprilTagNode::processImages, this));
      }
    }
  }

  return result;
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagNode)
