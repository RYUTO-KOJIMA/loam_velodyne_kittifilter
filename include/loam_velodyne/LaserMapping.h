
// LaserMapping.h

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

#ifndef LOAM_LASER_MAPPING_H
#define LOAM_LASER_MAPPING_H

#include "loam_velodyne/BasicLaserMapping.h"
#include "loam_velodyne/Common.h"

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

namespace loam {

/** \brief Implementation of the LOAM laser mapping component. */
class LaserMapping : public BasicLaserMapping
{
public:
    LaserMapping(const float scanPeriod = 0.1,
                 const std::size_t maxIterations = 10);

    /** \brief Setup component in active mode.
     *
     * @param node The ROS node handle
     * @param privateNode The private ROS node handle
     */
    virtual bool setup(ros::NodeHandle& node, ros::NodeHandle& privateNode);

    /** \brief Handler method for a new last corner cloud.
     *
     * @param cornerPointsLastMsg The new last corner cloud message
     */
    void laserCloudCornerLastHandler(
        const sensor_msgs::PointCloud2ConstPtr& cornerPointsLastMsg);

    /** \brief Handler method for a new last surface cloud.
     *
     * @param surfacePointsLastMsg the new last surface cloud message
     */
    void laserCloudSurfLastHandler(
        const sensor_msgs::PointCloud2ConstPtr& surfacePointsLastMsg);

    /** \brief Handler method for a new full resolution cloud.
     *
     * @param laserCloudFullResMsg The new full resolution cloud message
     */
    void laserCloudFullResHandler(
        const sensor_msgs::PointCloud2ConstPtr& laserCloudFullResMsg);

    /** \brief Handler method for a new laser odometry.
     *
     * @param laserOdometry The new laser odometry message
     */
    void laserOdometryHandler(
        const nav_msgs::Odometry::ConstPtr& laserOdometry);

    /** \brief Handler method for IMU messages.
     *
     * @param imuIn The new IMU message
     */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn);

    /** \brief Process incoming messages in a loop until shutdown
     * (used in active mode). */
    void spin();

    /** \brief Try to process buffered data. */
    void process();

protected:
    /** \brief Reset flags, etc. */
    void reset();

    /** \brief Check if all required information for a new processing step
     * is available. */
    bool hasNewData();

    /** \brief Publish the current result via the respective topics. */
    void publishResult();

private:
    // Time of current last corner cloud
    ros::Time _timeLaserCloudCornerLast;
    // Time of current last surface cloud
    ros::Time _timeLaserCloudSurfLast;
    // Time of current full resolution cloud
    ros::Time _timeLaserCloudFullRes;
    // Time of current laser odometry
    ros::Time _timeLaserOdometry;

    // Flag if a new last corner cloud has been received
    bool _newLaserCloudCornerLast;
    // Flag if a new last surface cloud has been received
    bool _newLaserCloudSurfLast;
    // Flag if a new full resolution cloud has been received
    bool _newLaserCloudFullRes;
    // Flag if a new laser odometry has been received
    bool _newLaserOdometry;

    // Mapping odometry message
    nav_msgs::Odometry _odomAftMapped;
    // Mapping odometry transformation
    tf::StampedTransform _aftMappedTrans;

    // Map cloud message publisher
    ros::Publisher _pubLaserCloudSurround;
    // Current full resolution cloud message publisher
    ros::Publisher _pubLaserCloudFullRes;
    // Mapping odometry publisher
    ros::Publisher _pubOdomAftMapped;
    // Mapping odometry transform broadcaster
    tf::TransformBroadcaster _tfBroadcaster;
    // Metrics message publisher
    ros::Publisher _pubMetrics;

    // Last corner cloud message subscriber
    ros::Subscriber _subLaserCloudCornerLast;
    // Last surface cloud message subscriber
    ros::Subscriber _subLaserCloudSurfLast;
    // Full resolution cloud message subscriber
    ros::Subscriber _subLaserCloudFullRes;
    // Laser odometry message subscriber
    ros::Subscriber _subLaserOdometry;
    // IMU message subscriber
    ros::Subscriber _subImu;

    // Flag enabled if the new point cloud is not processed by the node
    bool _pointCloudUnprocessed;
    // Number of the dropped point clouds
    int _numOfDroppedPointClouds;
};

} // namespace loam

#endif //LOAM_LASER_MAPPING_H
