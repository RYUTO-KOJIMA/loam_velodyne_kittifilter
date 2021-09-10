
// ScanRegistration.cpp

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#include "loam_velodyne/ScanRegistration.h"
#include "loam_velodyne/MathUtils.h"

#include <tf/transform_datatypes.h>

using namespace loam_velodyne;

namespace loam {

bool ScanRegistration::parseParams(
    const ros::NodeHandle& nh, RegistrationParams& configOut) 
{
    int intVal = 0;
    float floatVal = 0.0f;

    if (nh.getParam("scanPeriod", floatVal)) {
        if (floatVal <= 0) {
            ROS_ERROR("Invalid scanPeriod: %f (expected > 0)", floatVal);
            return false;
        } else {
            configOut.scanPeriod = floatVal;
            ROS_INFO("Set scanPeriod: %g", floatVal);
        }
    }

    if (nh.getParam("imuHistorySize", intVal)) {
        if (intVal < 1) {
            ROS_ERROR("Invalid imuHistorySize: %d (expected >= 1)", intVal);
            return false;
        } else {
            configOut.imuHistorySize = intVal;
            ROS_INFO("Set imuHistorySize: %d", intVal);
        }
    }

    if (nh.getParam("featureRegions", intVal)) {
        if (intVal < 1) {
            ROS_ERROR("Invalid featureRegions: %d (expected >= 1)", intVal);
            return false;
        } else {
            configOut.nFeatureRegions = intVal;
            ROS_INFO("Set nFeatureRegions: %d", intVal);
        }
    }

    if (nh.getParam("curvatureRegion", intVal)) {
        if (intVal < 1) {
            ROS_ERROR("Invalid curvatureRegion: %d (expected >= 1)", intVal);
            return false;
        } else {
            configOut.curvatureRegion = intVal;
            ROS_INFO("Set curvatureRegion: +/- %d", intVal);
        }
    }

    if (nh.getParam("maxCornerSharp", intVal)) {
        if (intVal < 1) {
            ROS_ERROR("Invalid maxCornerSharp: %d (expected >= 1)", intVal);
            return false;
        } else {
            configOut.maxCornerSharp = intVal;
            configOut.maxCornerLessSharp = 10 * intVal;
            ROS_INFO("Set maxCornerSharp / less sharp: %d / %d",
                     intVal, configOut.maxCornerLessSharp);
        }
    }

    if (nh.getParam("maxCornerLessSharp", intVal)) {
        if (intVal < configOut.maxCornerSharp) {
            ROS_ERROR("Invalid maxCornerLessSharp: %d (expected >= %d)",
                      intVal, configOut.maxCornerSharp);
            return false;
        } else {
            configOut.maxCornerLessSharp = intVal;
            ROS_INFO("Set maxCornerLessSharp: %d", intVal);
        }
    }

    if (nh.getParam("maxSurfaceFlat", intVal)) {
        if (intVal < 1) {
            ROS_ERROR("Invalid maxSurfaceFlat: %d (expected >= 1)", intVal);
            return false;
        } else {
            configOut.maxSurfaceFlat = intVal;
            ROS_INFO("Set maxSurfaceFlat: %d", intVal);
        }
    }

    if (nh.getParam("surfaceCurvatureThreshold", floatVal)) {
        if (floatVal < 0.001) {
            ROS_ERROR("Invalid surfaceCurvatureThreshold: "
                      "%f (expected >= 0.001)", floatVal);
            return false;
        } else {
            configOut.surfaceCurvatureThreshold = floatVal;
            ROS_INFO("Set surfaceCurvatureThreshold: %g", floatVal);
        }
    }

    if (nh.getParam("lessFlatFilterSize", floatVal)) {
        if (floatVal < 0.001) {
            ROS_ERROR("Invalid lessFlatFilterSize: "
                      "%f (expected >= 0.001)", floatVal);
            return false;
        } else {
            configOut.lessFlatFilterSize = floatVal;
            ROS_INFO("Set lessFlatFilterSize: %g", floatVal);
        }
    }

    if (!nh.getParam("publishMetrics", this->_metricsEnabled))
        this->_metricsEnabled = false;

    return true;
}

bool ScanRegistration::setupROS(
    ros::NodeHandle& node, ros::NodeHandle& privateNode,
    RegistrationParams& configOut)
{
    if (!this->parseParams(privateNode, configOut))
        return false;

    // Subscribe IMU topic
    this->_subImu = node.subscribe<sensor_msgs::Imu>(
        "/imu/data", 50, &ScanRegistration::handleIMUMessage, this);

    // Advertise scan registration topics
    this->_pubLaserCloud = node.advertise<sensor_msgs::PointCloud2>(
        "/velodyne_cloud_2", 2);
    this->_pubCornerPointsSharp = node.advertise<sensor_msgs::PointCloud2>(
        "/laser_cloud_sharp", 2);
    this->_pubCornerPointsLessSharp = node.advertise<sensor_msgs::PointCloud2>(
        "/laser_cloud_less_sharp", 2);
    this->_pubSurfPointsFlat = node.advertise<sensor_msgs::PointCloud2>(
        "/laser_cloud_flat", 2);
    this->_pubSurfPointsLessFlat = node.advertise<sensor_msgs::PointCloud2>(
        "/laser_cloud_less_flat", 2);
    this->_pubImuTrans = node.advertise<sensor_msgs::PointCloud2>(
        "/imu_trans", 5);

    if (this->_metricsEnabled)
        this->_pubMetrics = node.advertise<ScanRegistrationMetrics>(
            "/scan_registration_metrics", 2);

    return true;
}

void ScanRegistration::handleIMUMessage(const sensor_msgs::Imu::ConstPtr& imuIn)
{
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(imuIn->orientation, orientation);

    double roll;
    double pitch;
    double yaw;
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    // Consider the acceleration from gravity
    // Convert the gravity vector (0, 0, -9.81) in the world coordinate frame
    // to the IMU coordinate frame and swap the axes, as (x, y, z) axes in LOAM
    // corresponds to (y, z, x) axes in ROS coordinate systems
    Vector3 acc;
    acc.x() = static_cast<float>(imuIn->linear_acceleration.y
                                 - std::sin(roll) * std::cos(pitch) * 9.81);
    acc.y() = static_cast<float>(imuIn->linear_acceleration.z
                                 - std::cos(roll) * std::cos(pitch) * 9.81);
    acc.z() = static_cast<float>(imuIn->linear_acceleration.x
                                 + std::sin(pitch) * 9.81);

    IMUState newState;
    newState.stamp = fromROSTime(imuIn->header.stamp);
    newState.roll = roll;
    newState.pitch = pitch;
    newState.yaw = yaw;
    newState.acceleration = acc;

    this->updateIMUData(acc, newState);
}

void ScanRegistration::publishResult()
{
    const auto sweepStartTime = toROSTime(this->sweepStart());

    // Publish full resolution and feature point clouds
    publishCloudMsg(this->_pubLaserCloud,
                    this->laserCloud(), sweepStartTime, "/camera");
    publishCloudMsg(this->_pubCornerPointsSharp,
                    this->cornerPointsSharp(), sweepStartTime, "/camera");
    publishCloudMsg(this->_pubCornerPointsLessSharp,
                    this->cornerPointsLessSharp(), sweepStartTime, "/camera");
    publishCloudMsg(this->_pubSurfPointsFlat,
                    this->surfacePointsFlat(), sweepStartTime, "/camera");
    publishCloudMsg(this->_pubSurfPointsLessFlat,
                    this->surfacePointsLessFlat(), sweepStartTime, "/camera");

    // Publish corresponding IMU transformation information
    publishCloudMsg(this->_pubImuTrans,
                    this->imuTransform(), sweepStartTime, "/camera");

    // Set the timestamp of the metrics message
    this->_metricsMsg.stamp = ros::Time::now();

    // Publish the metrics message
    if (this->_metricsEnabled)
        this->_pubMetrics.publish(this->_metricsMsg);
}

} // namespace loam
