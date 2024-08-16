/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <Eigen/Core>

#include "nvapriltags_ros2/msg/april_tag_detection.hpp"
#include "nvapriltags_ros2/msg/april_tag_detection_array.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include "utils/parameters.hpp"

class AprilTagNode : public rclcpp::Node {
public:
    AprilTagNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

private:
    void onCameraFrame(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img);
   
    void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg_ci);

    void controlProcessing(const std_srvs::srv::SetBool::Request::SharedPtr request,
                         std_srvs::srv::SetBool::Response::SharedPtr response);

    void processImages();

    rcl_interfaces::msg::SetParametersResult parameters_cb(const std::vector<rclcpp::Parameter>& parameters);

    std::string tag_family_;
    double tag_edge_size_;
    int max_tags_;
    int throttle_interval_ms_ = 100;
    bool enable_processing_ = true;
    bool system_initialized_ = false;

    // const image_transport::CameraSubscriber sub_cam_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_cam_info_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_cam_;
    rclcpp::CallbackGroup::SharedPtr detection_callback_group_;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> detection_exec_;
    rclcpp::TimerBase::SharedPtr processing_timer_;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_enable_processing_;
    const rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr pub_tf_;
    const rclcpp::Publisher<nvapriltags_ros2::msg::AprilTagDetectionArray>::SharedPtr pub_detections_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr saved_cam_info_;

    NodeParamManager param_manager_;
    OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;

    struct AprilTagsImpl;
    std::unique_ptr<AprilTagsImpl> impl_;
};
