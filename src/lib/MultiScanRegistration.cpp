
// MultiScanRegistration.cpp

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

#include "loam_velodyne/MultiScanRegistration.h"
#include "math_utils.h"

#include <pcl_conversions/pcl_conversions.h>

namespace loam {

MultiScanMapper::MultiScanMapper(const float lowerBound,
                                 const float upperBound,
                                 const std::uint16_t nScanRings) :
    _lowerBound(lowerBound),
    _upperBound(upperBound),
    _nScanRings(nScanRings),
    _factor((nScanRings - 1) / (upperBound - lowerBound))
{
}

void MultiScanMapper::set(const float lowerBound,
                          const float upperBound,
                          const std::uint16_t nScanRings)
{
    this->_lowerBound = lowerBound;
    this->_upperBound = upperBound;
    this->_nScanRings = nScanRings;
    this->_factor = (nScanRings - 1) / (upperBound - lowerBound);
}

MultiScanRegistration::MultiScanRegistration(
    const MultiScanMapper& scanMapper) :
    _scanMapper(scanMapper)
{
}

bool MultiScanRegistration::setup(
    ros::NodeHandle& node, ros::NodeHandle& privateNode)
{
    RegistrationParams config;
    if (!this->setupROS(node, privateNode, config))
        return false;

    this->configure(config);
    return true;
}

bool MultiScanRegistration::setupROS(
    ros::NodeHandle& node, ros::NodeHandle& privateNode,
    RegistrationParams& configOut)
{
    if (!ScanRegistration::setupROS(node, privateNode, configOut))
        return false;

    // Fetch scan mapping params
    std::string lidarName;

    if (privateNode.getParam("lidar", lidarName)) {
        if (lidarName == "VLP-16") {
            this->_scanMapper = MultiScanMapper::Velodyne_VLP_16();
        } else if (lidarName == "HDL-32") {
            this->_scanMapper = MultiScanMapper::Velodyne_HDL_32();
        } else if (lidarName == "HDL-64E") {
            this->_scanMapper = MultiScanMapper::Velodyne_HDL_64E();
        } else {
            ROS_ERROR("Invalid lidar parameter: %s "
                      "(only \"VLP-16\", \"HDL-32\" and "
                      " \"HDL-64E\" are supported)", lidarName.c_str());
            return false;
        }

        ROS_INFO("Set %s scan mapper.", lidarName.c_str());
        if (!privateNode.hasParam("scanPeriod")) {
            configOut.scanPeriod = 0.1;
            ROS_INFO("Set scanPeriod: %f", configOut.scanPeriod);
        }
    } else {
        float vAngleMin;
        float vAngleMax;
        int nScanRings;

        if (privateNode.getParam("minVerticalAngle", vAngleMin) &&
            privateNode.getParam("maxVerticalAngle", vAngleMax) &&
            privateNode.getParam("nScanRings", nScanRings)) {
            if (vAngleMin >= vAngleMax) {
                ROS_ERROR("Invalid vertical range (min >= max)");
                return false;
            } else if (nScanRings < 2) {
                ROS_ERROR("Invalid number of scan rings (n < 2)");
                return false;
            }

            this->_scanMapper.set(vAngleMin, vAngleMax, nScanRings);
            ROS_INFO("Set linear scan mapper from %g to %g degrees "
                     "with %d scan rings.", vAngleMin, vAngleMax, nScanRings);
        }
    }

    // Subscribe to input cloud topic
    this->_subLaserCloud = node.subscribe<sensor_msgs::PointCloud2>(
        "/multi_scan_points", 2,
        &MultiScanRegistration::handleCloudMessage, this);

    return true;
}

void MultiScanRegistration::handleCloudMessage(
    const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
    if (--this->_systemDelay >= 0)
        return;

    // Fetch new input cloud
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

    this->process(laserCloudIn, fromROSTime(laserCloudMsg->header.stamp));
}

void MultiScanRegistration::process(
    const pcl::PointCloud<pcl::PointXYZ>& laserCloudIn, const Time& scanTime)
{
    const size_t cloudSize = laserCloudIn.size();

    // Determine scan start and end orientations
    float startOri = -std::atan2(laserCloudIn[0].y, laserCloudIn[0].x);
    float endOri = -std::atan2(laserCloudIn[cloudSize - 1].y,
                               laserCloudIn[cloudSize - 1].x) + 2.0f * M_PI;

    // `M_PI` <= `endOri - startOri` <= `3 * M_PI` holds
    if (endOri - startOri > 3.0f * M_PI)
        endOri -= 2.0f * M_PI;
    else if (endOri - startOri < M_PI)
        endOri += 2.0f * M_PI;

    bool halfPassed = false;
    pcl::PointXYZI point;

    // Clear all scanline points
    this->_laserCloudScans.resize(this->_scanMapper.getNumberOfScanRings());
    std::for_each(this->_laserCloudScans.begin(), this->_laserCloudScans.end(),
                  [](pcl::PointCloud<pcl::PointXYZI>& v) { v.clear(); });

    // Extract valid points from input cloud
    for (int i = 0; i < cloudSize; ++i) {
        // Swap the axes, as (x, y, z) axes in LOAM corresponds to
        // (y, z, x) axes in ROS coordinate systems
        point.x = laserCloudIn[i].y;
        point.y = laserCloudIn[i].z;
        point.z = laserCloudIn[i].x;

        // Skip NaN and Inf valued points
        if (!std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            !std::isfinite(point.z))
            continue;

        // Skip zero valued points
        if (calcSquaredPointDistance(point) < 0.0001)
            continue;

        // Compute vertical angle and scan ID (ring ID) of the point
        const float angle = std::atan(
            point.y / std::sqrt(point.x * point.x + point.z * point.z));
        const int scanID = this->_scanMapper.getRingForAngle(angle);

        // Skip points with invalid ring ID
        if (scanID >= this->_scanMapper.getNumberOfScanRings() || scanID < 0)
            continue;

        // Compute horizontal angle of the point
        float ori = -std::atan2(point.x, point.z);
        if (!halfPassed) {
            // `-M_PI / 2` <= `ori - startOri` <= `3 * M_PI / 2` holds
            if (ori < startOri - M_PI / 2.0f)
                ori += 2.0f * M_PI;
            else if (ori > startOri + M_PI * 3.0f / 2.0f)
                ori -= 2.0f * M_PI;

            if (ori - startOri > M_PI)
                halfPassed = true;
        } else {
            ori += 2 * M_PI;

            // `-3 * M_PI / 2` <= `ori - endOri` <= `M_PI / 2` holds
            if (ori < endOri - M_PI * 3.0f / 2.0f)
                ori += 2.0f * M_PI;
            else if (ori > endOri + M_PI / 2.0f)
                ori -= 2.0f * M_PI;
        }

        // Compute relative scan time based on point orientation
        const float relTime = this->config().scanPeriod
                              * (ori - startOri) / (endOri - startOri);
        point.intensity = scanID + relTime;

        // Project the point to the start of the sweep using IMU data
        this->projectPointToStartOfSweep(point, relTime);

        this->_laserCloudScans[scanID].push_back(point);
    }

    this->processScanlines(scanTime, this->_laserCloudScans);
    this->publishResult();
}

} // namespace loam
