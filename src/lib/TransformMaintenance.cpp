
// TransformMaintenance.cpp

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

#include "loam_velodyne/TransformMaintenance.h"

namespace loam {

TransformMaintenance::TransformMaintenance()
{
    // Initialize odometry and odometry tf messages
    this->_laserOdometry2.header.frame_id = "camera_init";
    this->_laserOdometry2.child_frame_id = "/camera";

    this->_laserOdometryTrans2.frame_id_ = "camera_init";
    this->_laserOdometryTrans2.child_frame_id_ = "/camera";
}

bool TransformMaintenance::setup(
    ros::NodeHandle &node, ros::NodeHandle &privateNode)
{
    // Advertise integrated laser odometry topic
    this->_pubLaserOdometry2 = node.advertise<nav_msgs::Odometry>(
        "/integrated_to_init", 5);

    // Subscribe to laser odometry and mapping odometry topics
    this->_subLaserOdometry = node.subscribe<nav_msgs::Odometry>(
        "/laser_odom_to_init", 5,
        &TransformMaintenance::laserOdometryHandler, this);
    this->_subOdomAftMapped = node.subscribe<nav_msgs::Odometry>(
        "/aft_mapped_to_init", 5,
        &TransformMaintenance::odomAftMappedHandler, this);

    return true;
}

void TransformMaintenance::laserOdometryHandler(
    const nav_msgs::Odometry::ConstPtr& laserOdometry)
{
    double roll;
    double pitch;
    double yaw;
    geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w))
        .getRPY(roll, pitch, yaw);

    // Swap and reverse the sign of Euler angles as in LaserOdometry
    this->updateOdometry(-pitch, -yaw, roll,
                         laserOdometry->pose.pose.position.x,
                         laserOdometry->pose.pose.position.y,
                         laserOdometry->pose.pose.position.z);

    // Combine the poses obtained from odometry and mapping
    this->transformAssociateToMap();

    geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
        this->transformMapped()[2], -this->transformMapped()[0],
        -this->transformMapped()[1]);

    // Swap and reverse the sign of Euler angles as in LaserOdometry
    this->_laserOdometry2.header.stamp = laserOdometry->header.stamp;
    this->_laserOdometry2.pose.pose.orientation.x = -geoQuat.y;
    this->_laserOdometry2.pose.pose.orientation.y = -geoQuat.z;
    this->_laserOdometry2.pose.pose.orientation.z = geoQuat.x;
    this->_laserOdometry2.pose.pose.orientation.w = geoQuat.w;
    this->_laserOdometry2.pose.pose.position.x = this->transformMapped()[3];
    this->_laserOdometry2.pose.pose.position.y = this->transformMapped()[4];
    this->_laserOdometry2.pose.pose.position.z = this->transformMapped()[5];
    this->_pubLaserOdometry2.publish(this->_laserOdometry2);

    this->_laserOdometryTrans2.stamp_ = laserOdometry->header.stamp;
    this->_laserOdometryTrans2.setRotation(
        tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    this->_laserOdometryTrans2.setOrigin(
        tf::Vector3(this->transformMapped()[3], this->transformMapped()[4],
                    this->transformMapped()[5]));
    this->_tfBroadcaster2.sendTransform(this->_laserOdometryTrans2);
}

void TransformMaintenance::odomAftMappedHandler(
    const nav_msgs::Odometry::ConstPtr& odomAftMapped)
{
    double roll;
    double pitch;
    double yaw;
    geometry_msgs::Quaternion geoQuat = odomAftMapped->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w))
        .getRPY(roll, pitch, yaw);

    // Swap and reverse the sign of Euler angles as in LaserOdometry
    this->updateMappingTransform(
        -pitch, -yaw, roll,
        odomAftMapped->pose.pose.position.x,
        odomAftMapped->pose.pose.position.y,
        odomAftMapped->pose.pose.position.z,
        odomAftMapped->twist.twist.angular.x,
        odomAftMapped->twist.twist.angular.y,
        odomAftMapped->twist.twist.angular.z,
        odomAftMapped->twist.twist.linear.x,
        odomAftMapped->twist.twist.linear.y,
        odomAftMapped->twist.twist.linear.z);
}

} // end namespace loam
