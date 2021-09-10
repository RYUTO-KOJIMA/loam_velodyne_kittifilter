
// LaserOdometry.cpp

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

#include <pcl/filters/filter.h>

#include "loam_velodyne/LaserOdometry.h"
#include "loam_velodyne/Common.h"
#include "loam_velodyne/MathUtils.h"

using namespace loam_velodyne;

namespace loam {

LaserOdometry::LaserOdometry(float scanPeriod,
                             std::uint16_t ioRatio,
                             std::size_t maxIterations) :
    BasicLaserOdometry(scanPeriod, maxIterations),
    _ioRatio(ioRatio)
{
    // Initialize odometry and odometry tf messages
    this->_laserOdometryMsg.header.frame_id = "camera_init";
    this->_laserOdometryMsg.child_frame_id = "/laser_odom";

    this->_laserOdometryTrans.frame_id_ = "camera_init";
    this->_laserOdometryTrans.child_frame_id_ = "/laser_odom";
}

bool LaserOdometry::setup(ros::NodeHandle& node, ros::NodeHandle& privateNode)
{
    // Fetch laser odometry parameters
    float fParam;
    int iParam;

    if (privateNode.getParam("scanPeriod", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid scanPeriod parameter: %f "
                      "(expected > 0)", fParam);
            return false;
        } else {
            this->setScanPeriod(fParam);
            ROS_INFO("Set scanPeriod: %g", fParam);
        }
    }

    if (privateNode.getParam("ioRatio", iParam)) {
        if (iParam < 1) {
            ROS_ERROR("Invalid ioRatio parameter: %d (expected > 0)", iParam);
            return false;
        } else {
            this->_ioRatio = iParam;
            ROS_INFO("Set ioRatio: %d", iParam);
        }
    }

    if (privateNode.getParam("maxIterations", iParam)) {
        if (iParam < 1) {
            ROS_ERROR("Invalid maxIterations parameter: %d "
                      "(expected > 0)", iParam);
            return false;
        } else {
            this->setMaxIterations(iParam);
            ROS_INFO("Set maxIterations: %d", iParam);
        }
    }

    if (privateNode.getParam("deltaTAbort", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid deltaTAbort parameter: %f "
                      "(expected > 0)", fParam);
            return false;
        } else {
            this->setDeltaTAbort(fParam);
            ROS_INFO("Set deltaTAbort: %g", fParam);
        }
    }

    if (privateNode.getParam("deltaRAbort", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid deltaRAbort parameter: %f "
                      "(expected > 0)", fParam);
            return false;
        } else {
            this->setDeltaRAbort(fParam);
            ROS_INFO("Set deltaRAbort: %g", fParam);
        }
    }

    if (!privateNode.getParam("publishMetrics", this->_metricsEnabled))
        this->_metricsEnabled = false;

    // Advertise laser odometry topics
    this->_pubLaserCloudCornerLast = node.advertise<sensor_msgs::PointCloud2>(
        "/laser_cloud_corner_last", 2);
    this->_pubLaserCloudSurfLast = node.advertise<sensor_msgs::PointCloud2>(
        "/laser_cloud_surf_last", 2);
    this->_pubLaserCloudFullRes = node.advertise<sensor_msgs::PointCloud2>(
        "/velodyne_cloud_3", 2);
    this->_pubLaserOdometry = node.advertise<nav_msgs::Odometry>(
        "/laser_odom_to_init", 5);

    // Subscribe scan registration topics
    this->_subCornerPointsSharp = node.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_sharp", 2,
        &LaserOdometry::laserCloudSharpHandler, this);
    this->_subCornerPointsLessSharp = node.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_less_sharp", 2,
        &LaserOdometry::laserCloudLessSharpHandler, this);
    this->_subSurfPointsFlat = node.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_flat", 2,
        &LaserOdometry::laserCloudFlatHandler, this);
    this->_subSurfPointsLessFlat = node.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_less_flat", 2,
        &LaserOdometry::laserCloudLessFlatHandler, this);
    this->_subLaserCloudFullRes = node.subscribe<sensor_msgs::PointCloud2>(
        "/velodyne_cloud_2", 2,
        &LaserOdometry::laserCloudFullResHandler, this);
    this->_subImuTrans = node.subscribe<sensor_msgs::PointCloud2>(
        "/imu_trans", 5,
        &LaserOdometry::imuTransHandler, this);

    if (this->_metricsEnabled)
        this->_pubMetrics = node.advertise<LaserOdometryMetrics>(
            "/laser_odometry_metrics", 2);

    return true;
}

void LaserOdometry::reset()
{
    this->_newCornerPointsSharp = false;
    this->_newCornerPointsLessSharp = false;
    this->_newSurfPointsFlat = false;
    this->_newSurfPointsLessFlat = false;
    this->_newLaserCloudFullRes = false;
    this->_newImuTrans = false;
}

void LaserOdometry::laserCloudSharpHandler(
    const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharpMsg)
{
    this->_timeCornerPointsSharp = cornerPointsSharpMsg->header.stamp;

    this->cornerPointsSharp()->clear();
    pcl::fromROSMsg(*cornerPointsSharpMsg, *this->cornerPointsSharp());
    removeNaNFromPointCloud<pcl::PointXYZI>(this->cornerPointsSharp());
    this->_newCornerPointsSharp = true;
}

void LaserOdometry::laserCloudLessSharpHandler(
    const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharpMsg)
{
    this->_timeCornerPointsLessSharp = cornerPointsLessSharpMsg->header.stamp;

    this->cornerPointsLessSharp()->clear();
    pcl::fromROSMsg(*cornerPointsLessSharpMsg, *this->cornerPointsLessSharp());
    removeNaNFromPointCloud<pcl::PointXYZI>(this->cornerPointsLessSharp());
    this->_newCornerPointsLessSharp = true;
}

void LaserOdometry::laserCloudFlatHandler(
    const sensor_msgs::PointCloud2ConstPtr& surfPointsFlatMsg)
{
    this->_timeSurfPointsFlat = surfPointsFlatMsg->header.stamp;

    this->surfPointsFlat()->clear();
    pcl::fromROSMsg(*surfPointsFlatMsg, *this->surfPointsFlat());
    removeNaNFromPointCloud<pcl::PointXYZI>(this->surfPointsFlat());
    this->_newSurfPointsFlat = true;
}

void LaserOdometry::laserCloudLessFlatHandler(
    const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlatMsg)
{
    this->_timeSurfPointsLessFlat = surfPointsLessFlatMsg->header.stamp;

    this->surfPointsLessFlat()->clear();
    pcl::fromROSMsg(*surfPointsLessFlatMsg, *this->surfPointsLessFlat());
    removeNaNFromPointCloud<pcl::PointXYZI>(this->surfPointsLessFlat());
    this->_newSurfPointsLessFlat = true;
}

void LaserOdometry::laserCloudFullResHandler(
    const sensor_msgs::PointCloud2ConstPtr& laserCloudFullResMsg)
{
    this->_timeLaserCloudFullRes = laserCloudFullResMsg->header.stamp;

    this->laserCloud()->clear();
    pcl::fromROSMsg(*laserCloudFullResMsg, *this->laserCloud());
    removeNaNFromPointCloud<pcl::PointXYZI>(this->laserCloud());
    this->_newLaserCloudFullRes = true;
}

void LaserOdometry::imuTransHandler(
    const sensor_msgs::PointCloud2ConstPtr& imuTransMsg)
{
    this->_timeImuTrans = imuTransMsg->header.stamp;

    pcl::PointCloud<pcl::PointXYZ> imuTrans;
    pcl::fromROSMsg(*imuTransMsg, imuTrans);
    this->updateIMU(imuTrans);
    this->_newImuTrans = true;
}

void LaserOdometry::spin()
{
    ros::Rate rate(100);
    bool status = ros::ok();

    // Loop until shutdown
    while (status) {
        ros::spinOnce();

        // Try processing new data
        this->process();

        status = ros::ok();
        rate.sleep();
    }
}

bool LaserOdometry::hasNewData()
{
    // Compute the differences between the timestamp of the most recent
    // less flat point cloud and timestamps of the other messages
    const auto diffSharpPoints =
        this->_timeCornerPointsSharp - this->_timeSurfPointsLessFlat;
    const auto diffLessSharpPoints =
        this->_timeCornerPointsLessSharp - this->_timeSurfPointsLessFlat;
    const auto diffFlatPoints =
        this->_timeSurfPointsFlat - this->_timeSurfPointsLessFlat;
    const auto diffFullResPoints =
        this->_timeLaserCloudFullRes - this->_timeSurfPointsLessFlat;
    const auto diffImuTrans =
        this->_timeImuTrans - this->_timeSurfPointsLessFlat;

    return this->_newCornerPointsSharp &&
           this->_newCornerPointsLessSharp &&
           this->_newSurfPointsFlat &&
           this->_newSurfPointsLessFlat &&
           this->_newLaserCloudFullRes &&
           this->_newImuTrans &&
           std::fabs(diffSharpPoints.toSec()) < 0.005 &&
           std::fabs(diffLessSharpPoints.toSec()) < 0.005 &&
           std::fabs(diffFlatPoints.toSec()) < 0.005 &&
           std::fabs(diffFullResPoints.toSec()) < 0.005 &&
           std::fabs(diffImuTrans.toSec()) < 0.005;
}

void LaserOdometry::process()
{
    if (!this->hasNewData())
        return;

    this->reset();
    this->clearMetricsMsg();
    BasicLaserOdometry::process();
    this->publishResult();
}

void LaserOdometry::publishResult()
{
    // ROS uses a right-handed coordinate system with x-axis forward,
    // y-axis left, and z-axis upward, while LOAM implementation uses
    // a right-handed coordinate system with z-axis forward, x-axis left, and
    // y-axis upward, i.e., ROS uses fixed XYZ Euler angles, while LOAM uses
    // fixed ZXY Euler angles

    // Compute (geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z) =
    // q_z(-rot_y) q_y(-rot_x) q_x(rot_z) using fixed XYZ Euler angles,
    // and q_y(rot_y) q_x(rot_x) q_z(rot_z) is easily constructed as
    // (geoQuat.w, -geoQuat.y, -geoQuat.z, -geoQuat.x)

    // Publish odometry transformations
    const geometry_msgs::Quaternion geoQuat =
        tf::createQuaternionMsgFromRollPitchYaw(
            this->transformSum().rot_z.rad(),
            -this->transformSum().rot_x.rad(),
            -this->transformSum().rot_y.rad());

    this->_laserOdometryMsg.header.stamp = this->_timeSurfPointsLessFlat;
    // (orientation.w, orientation.x, orientation.y, orientation.z) equals to
    // q_y(rot_y) q_x(rot_x) q_z(rot_z) which represents the rotation
    // quaternion of the optimized odometry pose `transformSum()`
    auto& poseMsg = this->_laserOdometryMsg.pose.pose;
    poseMsg.orientation.x = -geoQuat.y;
    poseMsg.orientation.y = -geoQuat.z;
    poseMsg.orientation.z = geoQuat.x;
    poseMsg.orientation.w = geoQuat.w;
    poseMsg.position.x = this->transformSum().pos.x();
    poseMsg.position.y = this->transformSum().pos.y();
    poseMsg.position.z = this->transformSum().pos.z();
    this->_pubLaserOdometry.publish(this->_laserOdometryMsg);

    this->_laserOdometryTrans.stamp_ = this->_timeSurfPointsLessFlat;
    // (geoQuat.w, -geoQuat.y, -geoQuat.z, geoQuat.x) equals to
    // q_y(rot_y) q_x(rot_x) q_z(rot_z)
    this->_laserOdometryTrans.setRotation(
        tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    this->_laserOdometryTrans.setOrigin(
        tf::Vector3(this->transformSum().pos.x(),
                    this->transformSum().pos.y(),
                    this->transformSum().pos.z()));
    this->_tfBroadcaster.sendTransform(this->_laserOdometryTrans);

    // Publish cloud results according to the input output ratio
    if (this->_ioRatio < 2 || this->frameCount() % this->_ioRatio == 1) {
        ros::Time sweepTime = this->_timeSurfPointsLessFlat;
        // `lastCornerCloud()` and `lastSurfaceCloud()` are already reprojected
        // to the end of the current sweep
        publishCloudMsg(this->_pubLaserCloudCornerLast,
                        *this->lastCornerCloud(), sweepTime, "/camera");
        publishCloudMsg(this->_pubLaserCloudSurfLast,
                        *this->lastSurfaceCloud(), sweepTime, "/camera");

        // `laserCloud()` is reprojected to the beginning of the current sweep,
        // and should be reprojected to the end of the current sweep
        this->transformToEnd(this->laserCloud());
        publishCloudMsg(this->_pubLaserCloudFullRes,
                        *this->laserCloud(), sweepTime, "/camera");

        // Collect the metrics
        this->_metricsMsg.point_cloud_stamp = sweepTime;
        this->_metricsMsg.num_of_full_res_points =
            this->laserCloud()->size();
        this->_metricsMsg.num_of_less_sharp_points =
            this->lastCornerCloud()->size();
        this->_metricsMsg.num_of_less_flat_points =
            this->lastSurfaceCloud()->size();
    }

    // Publish the metrics message
    if (this->_metricsEnabled)
        this->_pubMetrics.publish(this->_metricsMsg);
}

} // namespace loam
