
// LaserMapping.cpp

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

#include "loam_velodyne/LaserMapping.h"
#include "loam_velodyne/Common.h"

using namespace loam_velodyne;

namespace loam {

LaserMapping::LaserMapping(const float scanPeriod,
                           const std::size_t maxIterations) :
    _timeLaserCloudCornerLast(0.0),
    _timeLaserCloudSurfLast(0.0),
    _timeLaserCloudFullRes(0.0),
    _timeLaserOdometry(0.0),
    _newLaserCloudCornerLast(false),
    _newLaserCloudSurfLast(false),
    _newLaserCloudFullRes(false),
    _newLaserOdometry(false),
    _pointCloudUnprocessed(false),
    _numOfDroppedPointClouds(0)
{
    // Initialize mapping odometry and odometry tf messages
    this->_odomAftMapped.header.frame_id = "camera_init";
    this->_odomAftMapped.child_frame_id = "/aft_mapped";

    this->_aftMappedTrans.frame_id_ = "camera_init";
    this->_aftMappedTrans.child_frame_id_ = "/aft_mapped";
}

bool LaserMapping::setup(ros::NodeHandle& node, ros::NodeHandle& privateNode)
{
    // Fetch the node parameters
    auto getInt = [&privateNode](const char* key, int& value) {
        return privateNode.getParam(key, value); };
    auto getBool = [&privateNode](const char* key, bool& value) {
        return privateNode.getParam(key, value); };
    auto getFloat = [&privateNode](const char* key, float& value) {
        return privateNode.getParam(key, value); };

    float fParam;
    int iParam;
    bool boolValue;

    if (getFloat("scanPeriod", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid scanPeriod parameter: %f "
                      "(expected > 0, default: 0.1)", fParam);
            return false;
        } else {
            this->setScanPeriod(fParam);
        }
    }

    if (getInt("maxIterations", iParam)) {
        if (iParam < 1) {
            ROS_ERROR("Invalid maxIterations parameter: %d "
                      "(expected > 0, default: 10)", iParam);
            return false;
        } else {
            this->setMaxIterations(iParam);
        }
    }

    if (getFloat("deltaTAbort", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid deltaTAbort parameter: %f "
                      "(expected > 0, default: 0.05)", fParam);
            return false;
        } else {
            this->setDeltaTAbort(fParam);
        }
    }

    if (getFloat("deltaRAbort", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid deltaRAbort parameter: %f "
                      "(expected > 0, default: 0.05)", fParam);
            return false;
        } else {
            this->setDeltaRAbort(fParam);
        }
    }

    if (getFloat("cornerFilterSize", fParam)) {
        if (fParam < 0.001f) {
            ROS_ERROR("Invalid cornerFilterSize parameter: %f "
                      "(expected >= 0.001, default: 0.2)", fParam);
            return false;
        } else {
            this->downSizeFilterCorner().setLeafSize(fParam, fParam, fParam);
        }
    }

    if (getFloat("surfaceFilterSize", fParam)) {
        if (fParam < 0.001f) {
            ROS_ERROR("Invalid surfaceFilterSize parameter: %f "
                      "(expected >= 0.001, default: 0.4)", fParam);
            return false;
        } else {
            this->downSizeFilterSurf().setLeafSize(fParam, fParam, fParam);
        }
    }

    if (getFloat("mapFilterSize", fParam)) {
        if (fParam < 0.001f) {
            ROS_ERROR("Invalid mapFilterSize parameter: %f "
                      "(expected >= 0.001: default: 0.0)", fParam);
            return false;
        } else {
            this->downSizeFilterMap().setLeafSize(fParam, fParam, fParam);
        }
    }

    ROS_INFO("BasicLaserMapping::_scanPeriod: %g", this->scanPeriod());
    ROS_INFO("BasicLaserMapping::_maxIterations: %zu", this->maxIterations());
    ROS_INFO("BasicLaserMapping::_deltaTAbort: %g", this->deltaTAbort());
    ROS_INFO("BasicLaserMapping::_deltaRAbort: %g", this->deltaRAbort());
    ROS_INFO("BasicLaserMapping::_downSizeFilterCorner: (%g, %g, %g)",
             this->downSizeFilterCorner().getLeafSize().x(),
             this->downSizeFilterCorner().getLeafSize().y(),
             this->downSizeFilterCorner().getLeafSize().z());
    ROS_INFO("BasicLaserMapping::_downSizeFilterSurf: (%g, %g, %g)",
             this->downSizeFilterSurf().getLeafSize().x(),
             this->downSizeFilterSurf().getLeafSize().y(),
             this->downSizeFilterSurf().getLeafSize().z());
    ROS_INFO("BasicLaserMapping::_downSizeFilterMap: (%g, %g, %g)",
             this->downSizeFilterMap().getLeafSize().x(),
             this->downSizeFilterMap().getLeafSize().y(),
             this->downSizeFilterMap().getLeafSize().z());

    if (getFloat("residualScale", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid residualScale parameter: %f "
                      "(expected > 0, default: 1.0)", fParam);
            return false;
        } else {
            this->_residualScale = fParam;
        }
    }

    if (getFloat("eigenThresholdTrans", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid eigenThresholdTrans parameter: %f "
                      "(expected > 0, default: 100.0)", fParam);
            return false;
        } else {
            this->_eigenThresholdTrans = fParam;
        }
    }

    if (getFloat("eigenThresholdRot", fParam)) {
        if (fParam <= 0.0f) {
            ROS_ERROR("Invalid eigenThresholdRot parameter: %f "
                      "(expected > 0, default: 100.0)", fParam);
            return false;
        } else {
            this->_eigenThresholdRot = fParam;
        }
    }

    if (getFloat("sqDistThresholdCorner", fParam)) {
        if (fParam < 0.0f) {
            ROS_ERROR("Invalid sqDistThresholdCorner parameter: %f "
                      "(expected >= 0, default: 1.0)", fParam);
            return false;
        } else {
            this->_sqDistThresholdCorner = fParam;
        }
    }

    if (getFloat("weightDecayCorner", fParam)) {
        if (fParam < 0.0f) {
            ROS_ERROR("Invalid weightDecayCorner parameter: %f "
                      "(expected >= 0, default: 0.9)", fParam);
            return false;
        } else {
            this->_weightDecayCorner = fParam;
        }
    }

    if (getFloat("weightThresholdCorner", fParam)) {
        if (fParam < 0.0f) {
            ROS_ERROR("Invalid weightThresholdCorner parameter: %f "
                      "(expected >= 0, default: 0.1)", fParam);
            return false;
        } else {
            this->_weightThresholdCorner = fParam;
        }
    }

    if (getFloat("sqDistThresholdSurface", fParam)) {
        if (fParam < 0.0f) {
            ROS_ERROR("Invalid sqDistThresholdSurface parameter: %f "
                      "(expected >= 0, default: 1.0)", fParam);
            return false;
        } else {
            this->_sqDistThresholdSurface = fParam;
        }
    }

    if (getFloat("weightDecaySurface", fParam)) {
        if (fParam < 0.0f) {
            ROS_ERROR("Invalid weightDecaySurface parameter: %f "
                      "(expected >= 0, default: 0.9)", fParam);
            return false;
        } else {
            this->_weightDecaySurface = fParam;
        }
    }

    if (getFloat("weightThresholdSurface", fParam)) {
        if (fParam < 0.0f) {
            ROS_ERROR("Invalid weightThresholdSurface parameter: %f "
                      "(expected >= 0, default: 0.1)", fParam);
            return false;
        } else {
            this->_weightThresholdSurface = fParam;
        }
    }

    ROS_INFO("BasicLaserMapping::_residualScale: %g",
             this->_residualScale);
    ROS_INFO("BasicLaserMapping::_eigenThresholdTrans: %g",
             this->_eigenThresholdTrans);
    ROS_INFO("BasicLaserMapping::_eigenThresholdRot: %g",
             this->_eigenThresholdRot);
    ROS_INFO("BasicLaserMapping::_sqDistThresholdCorner: %g",
             this->_sqDistThresholdCorner);
    ROS_INFO("BasicLaserMapping::_weightDecayCorner: %g",
             this->_weightDecayCorner);
    ROS_INFO("BasicLaserMapping::_weightThresholdCorner: %g",
             this->_weightThresholdCorner);
    ROS_INFO("BasicLaserMapping::_sqDistThresholdSurface: %g",
             this->_sqDistThresholdSurface);
    ROS_INFO("BasicLaserMapping::_weightDecaySurface: %g",
             this->_weightDecaySurface);
    ROS_INFO("BasicLaserMapping::_weightThresholdSurface: %g",
             this->_weightThresholdSurface);

    if (getBool("publishFullPointCloud", boolValue))
        this->_fullPointCloudPublished = boolValue;
    else
        this->_fullPointCloudPublished = true;

    if (getBool("publishMetrics", boolValue))
        this->_metricsEnabled = boolValue;
    else
        this->_metricsEnabled = false;

    // Advertise laser mapping topics
    this->_pubLaserCloudSurround = node.advertise<sensor_msgs::PointCloud2>(
        "/laser_cloud_surround", 1);
    this->_pubOdomAftMapped = node.advertise<nav_msgs::Odometry>(
        "/aft_mapped_to_init", 5);

    if (this->_fullPointCloudPublished)
        this->_pubLaserCloudFullRes = node.advertise<sensor_msgs::PointCloud2>(
            "/velodyne_cloud_registered", 2);

    // Subscribe to laser odometry topics
    this->_subLaserCloudCornerLast = node.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_corner_last", 2,
        &LaserMapping::laserCloudCornerLastHandler, this);
    this->_subLaserCloudSurfLast = node.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_surf_last", 2,
        &LaserMapping::laserCloudSurfLastHandler, this);
    this->_subLaserOdometry = node.subscribe<nav_msgs::Odometry>(
        "/laser_odom_to_init", 5,
        &LaserMapping::laserOdometryHandler, this);

    if (this->_fullPointCloudPublished)
        this->_subLaserCloudFullRes = node.subscribe<sensor_msgs::PointCloud2>(
            "/velodyne_cloud_3", 2,
            &LaserMapping::laserCloudFullResHandler, this);

    // Subscribe to IMU topic
    this->_subImu = node.subscribe<sensor_msgs::Imu>(
        "/imu/data", 50, &LaserMapping::imuHandler, this);

    if (this->_metricsEnabled)
        this->_pubMetrics = node.advertise<LaserMappingMetrics>(
            "/laser_mapping_metrics", 2);

    return true;
}

void LaserMapping::laserCloudCornerLastHandler(
    const sensor_msgs::PointCloud2ConstPtr& cornerPointsLastMsg)
{
    //printf("LaserMapping: get msg !!\n");
    
    // Check that the previous point cloud is not processed by the node
    // If not processed, increment the number of the dropped point clouds
    if (this->_pointCloudUnprocessed) {
        ++this->_numOfDroppedPointClouds;
        ROS_WARN("Point cloud is dropped by LaserMapping node "
                 "(Number of the dropped point clouds: %d)",
                 this->_numOfDroppedPointClouds);
    }

    // Set the flag to indicate that the current point cloud is not processed
    this->_pointCloudUnprocessed = true;

    this->_timeLaserCloudCornerLast = cornerPointsLastMsg->header.stamp;
    this->laserCloudCornerLast().clear();
    pcl::fromROSMsg(*cornerPointsLastMsg, this->laserCloudCornerLast());
    this->_newLaserCloudCornerLast = true;
}

void LaserMapping::laserCloudSurfLastHandler(
    const sensor_msgs::PointCloud2ConstPtr& surfacePointsLastMsg)
{
    this->_timeLaserCloudSurfLast = surfacePointsLastMsg->header.stamp;
    this->laserCloudSurfLast().clear();
    pcl::fromROSMsg(*surfacePointsLastMsg, this->laserCloudSurfLast());
    this->_newLaserCloudSurfLast = true;
}

void LaserMapping::laserCloudFullResHandler(
    const sensor_msgs::PointCloud2ConstPtr& laserCloudFullResMsg)
{
    this->_timeLaserCloudFullRes = laserCloudFullResMsg->header.stamp;
    this->laserCloud().clear();
    pcl::fromROSMsg(*laserCloudFullResMsg, this->laserCloud());
    this->_newLaserCloudFullRes = true;
}

void LaserMapping::laserOdometryHandler(
    const nav_msgs::Odometry::ConstPtr& laserOdometry)
{
    this->_timeLaserOdometry = laserOdometry->header.stamp;

    double roll;
    double pitch;
    double yaw;
    geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w))
        .getRPY(roll, pitch, yaw);

    // Swap and reverse the sign of Euler angles so that `_transformSum` in
    // LaserMapping is the same as the `_transformSum` in LaserOdometry
    this->updateOdometry(-pitch, -yaw, roll,
                         laserOdometry->pose.pose.position.x,
                         laserOdometry->pose.pose.position.y,
                         laserOdometry->pose.pose.position.z);

    this->_newLaserOdometry = true;
}

void LaserMapping::imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
    // LaserMapping methods does not seem to use the yaw angle
    // (rotation on the vertical axis)
    double roll;
    double pitch;
    double yaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(imuIn->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    this->updateIMU({ fromROSTime(imuIn->header.stamp), roll, pitch });
}

void LaserMapping::spin()
{
    ros::Rate rate(100);
    bool status = ros::ok();

    while (status) {
        ros::spinOnce();

        // Try processing buffered data
        this->process();

        status = ros::ok();
        rate.sleep();
    }
}

void LaserMapping::reset()
{
    this->_newLaserCloudCornerLast = false;
    this->_newLaserCloudSurfLast = false;
    this->_newLaserCloudFullRes = false;
    this->_newLaserOdometry = false;
    this->_pointCloudUnprocessed = false;
}

bool LaserMapping::hasNewData()
{
    const auto diffCornerPoints =
        this->_timeLaserCloudCornerLast - this->_timeLaserOdometry;
    const auto diffSurfPoints =
        this->_timeLaserCloudSurfLast - this->_timeLaserOdometry;
    const auto diffFullResPoints =
        this->_timeLaserCloudFullRes - this->_timeLaserOdometry;

    const auto isFullResNew =
        !this->_fullPointCloudPublished ||
        (this->_newLaserCloudFullRes &&
         std::fabs(diffFullResPoints.toSec()) < 0.005);

    return this->_newLaserCloudCornerLast &&
           this->_newLaserCloudSurfLast &&
           this->_newLaserOdometry &&
           std::fabs(diffCornerPoints.toSec()) < 0.005 &&
           std::fabs(diffSurfPoints.toSec()) < 0.005 &&
           isFullResNew;
}

void LaserMapping::process()
{
    if (!this->hasNewData())
        return;

    this->reset();

    // Clear the metrics message
    this->clearMetricsMsg();

    if (!BasicLaserMapping::process(fromROSTime(this->_timeLaserOdometry)))
        return;

    this->publishResult();
}

void LaserMapping::publishResult()
{
    // Publish new map cloud according to the input output ratio
    if (this->hasFreshMap()) {
        publishCloudMsg(this->_pubLaserCloudSurround,
                        this->laserCloudSurroundDS(),
                        this->_timeLaserOdometry, "camera_init");

        // Collect the metrics
        this->_metricsMsg.num_of_surround_points =
            this->laserCloudSurroundDS().size();
        this->_metricsMsg.num_of_surround_points_before_ds =
            this->laserCloudSurround().size();
    }

    // Publish transformed full resolution input cloud
    if (this->_fullPointCloudPublished) {
        publishCloudMsg(this->_pubLaserCloudFullRes, this->laserCloud(),
                        this->_timeLaserOdometry, "camera_init");
        // Collect the metric
        this->_metricsMsg.num_of_full_res_points = this->laserCloud().size();
    }

    // Collect the metric
    this->_metricsMsg.point_cloud_stamp = this->_timeLaserOdometry;

    // Swap and reverse the sign of Euler angles as in LaserOdometry
    // Publish odometry after mapped transformations
    const geometry_msgs::Quaternion geoQuat =
        tf::createQuaternionMsgFromRollPitchYaw(
            this->transformAftMapped().rot_z.rad(),
            -this->transformAftMapped().rot_x.rad(),
            -this->transformAftMapped().rot_y.rad());

    this->_odomAftMapped.header.stamp = _timeLaserOdometry;
    auto& poseMsg = this->_odomAftMapped.pose.pose;
    poseMsg.orientation.x = -geoQuat.y;
    poseMsg.orientation.y = -geoQuat.z;
    poseMsg.orientation.z = geoQuat.x;
    poseMsg.orientation.w = geoQuat.w;
    poseMsg.position.x = this->transformAftMapped().pos.x();
    poseMsg.position.y = this->transformAftMapped().pos.y();
    poseMsg.position.z = this->transformAftMapped().pos.z();

    auto& twistMsg = this->_odomAftMapped.twist.twist;
    twistMsg.angular.x = this->transformBefMapped().rot_x.rad();
    twistMsg.angular.y = this->transformBefMapped().rot_y.rad();
    twistMsg.angular.z = this->transformBefMapped().rot_z.rad();
    twistMsg.linear.x = this->transformBefMapped().pos.x();
    twistMsg.linear.y = this->transformBefMapped().pos.y();
    twistMsg.linear.z = this->transformBefMapped().pos.z();
    this->_pubOdomAftMapped.publish(this->_odomAftMapped);

    this->_aftMappedTrans.stamp_ = this->_timeLaserOdometry;
    this->_aftMappedTrans.setRotation(
        tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    this->_aftMappedTrans.setOrigin(
        tf::Vector3(this->transformAftMapped().pos.x(),
                    this->transformAftMapped().pos.y(),
                    this->transformAftMapped().pos.z()));
    this->_tfBroadcaster.sendTransform(this->_aftMappedTrans);

    // Set the timestamp of the metrics message
    this->_metricsMsg.stamp = ros::Time::now();
    // Set the number of the dropped point clouds
    this->_metricsMsg.num_of_dropped_point_clouds =
        this->_numOfDroppedPointClouds;

    // Publish the metrics message
    if (this->_metricsEnabled)
        this->_pubMetrics.publish(this->_metricsMsg);
}

} // end namespace loam
