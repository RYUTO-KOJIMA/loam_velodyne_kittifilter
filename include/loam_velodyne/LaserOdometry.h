
// LaserOdometry.h

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

#ifndef LOAM_LASER_ODOMETRY_H
#define LOAM_LASER_ODOMETRY_H

#include "loam_velodyne/Twist.h"
#include "loam_velodyne/nanoflann_pcl.h"

#include <ros/node_handle.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "loam_velodyne/BasicLaserOdometry.h"

namespace loam {

/** \brief Implementation of the LOAM laser odometry component. */
class LaserOdometry : public BasicLaserOdometry
{
public:
    LaserOdometry(const float scanPeriod = 0.1,
                  const std::uint16_t ioRatio = 2,
                  const std::size_t maxIterations = 25);

    /** \brief Setup component.
     *
     * @param node The ROS node handle
     * @param privateNode The private ROS node handle
     */
    virtual bool setup(ros::NodeHandle& node, ros::NodeHandle& privateNode);

    /** \brief Handler method for a new sharp corner cloud.
     * @param cornerPointsSharpMsg The sharp corner cloud message */
    void laserCloudSharpHandler(
        const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharpMsg);

    /** \brief Handler method for a new less sharp corner cloud.
     * @param cornerPointsLessSharpMsg The less sharp corner cloud message */
    void laserCloudLessSharpHandler(
        const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharpMsg);

    /** \brief Handler method for a new flat surface cloud.
     * @param surfPointsFlatMsg The flat surface cloud message */
    void laserCloudFlatHandler(
        const sensor_msgs::PointCloud2ConstPtr& surfPointsFlatMsg);

    /** \brief Handler method for a new less flat surface cloud.
     * @param surfPointsLessFlatMsg The less flat surface cloud message */
    void laserCloudLessFlatHandler(
        const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlatMsg);

    /** \brief Handler method for a new full resolution cloud.
     * @param laserCloudFullResMsg The full resolution cloud message */
    void laserCloudFullResHandler(
        const sensor_msgs::PointCloud2ConstPtr& laserCloudFullResMsg);

    /** \brief Handler method for a new IMU transformation.
     * @param laserCloudFullResMsg The IMU transformation message */
    void imuTransHandler(const sensor_msgs::PointCloud2ConstPtr& imuTransMsg);

    /** \brief Process incoming messages in a loop until shutdown
     * (used in active mode). */
    void spin();

    /** \brief Try to process buffered data. */
    void process();

protected:
    /** \brief Reset flags, etc. */
    void reset();

    /** \brief Check if all required information for a new processing
     * step is available. */
    bool hasNewData();

    /** \brief Publish the current result via the respective topics. */
    void publishResult();

private:
    // Ratio of input to output frames
    std::uint16_t _ioRatio;

    // Time of current sharp corner cloud
    ros::Time _timeCornerPointsSharp;
    // Time of current less sharp corner cloud
    ros::Time _timeCornerPointsLessSharp;
    // Time of current flat surface cloud
    ros::Time _timeSurfPointsFlat;
    // Time of current less flat surface cloud
    ros::Time _timeSurfPointsLessFlat;
    // Time of current full resolution cloud
    ros::Time _timeLaserCloudFullRes;
    // Time of current IMU transformation information
    ros::Time _timeImuTrans;

    // Flag if a new sharp corner cloud has been received
    bool _newCornerPointsSharp;
    // Flag if a new less sharp corner cloud has been received
    bool _newCornerPointsLessSharp;
    // Flag if a new flat surface cloud has been received
    bool _newSurfPointsFlat;
    // Flag if a new less flat surface cloud has been received
    bool _newSurfPointsLessFlat;
    // Flag if a new full resolution cloud has been received
    bool _newLaserCloudFullRes;
    // Flag if a new IMU transformation information cloud has been received
    bool _newImuTrans;

    // Laser odometry message
    nav_msgs::Odometry _laserOdometryMsg;
    // Laser odometry transformation
    tf::StampedTransform _laserOdometryTrans;

    // Last corner cloud message publisher
    ros::Publisher _pubLaserCloudCornerLast;
    // Last surface cloud message publisher
    ros::Publisher _pubLaserCloudSurfLast;
    // Full resolution cloud message publisher
    ros::Publisher _pubLaserCloudFullRes;
    // Laser odometry publisher
    ros::Publisher _pubLaserOdometry;
    // Laser odometry transform broadcaster
    tf::TransformBroadcaster _tfBroadcaster;
    // Metrics message publisher
    ros::Publisher _pubMetrics;

    // Sharp corner cloud message subscriber
    ros::Subscriber _subCornerPointsSharp;
    // Less sharp corner cloud message subscriber
    ros::Subscriber _subCornerPointsLessSharp;
    // Flat surface cloud message subscriber
    ros::Subscriber _subSurfPointsFlat;
    // Less flat surface cloud message subscriber
    ros::Subscriber _subSurfPointsLessFlat;
    // Full resolution cloud message subscriber
    ros::Subscriber _subLaserCloudFullRes;
    // IMU transformation message subscriber
    ros::Subscriber _subImuTrans;

    // Flag enabled if the new point cloud is not processed by the node
    bool _pointCloudUnprocessed;
    // Number of the dropped point clouds
    int _numOfDroppedPointClouds;
};

} // namespace loam

#endif //LOAM_LASER_ODOMETRY_H
