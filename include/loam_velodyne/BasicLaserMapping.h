
// BasicLaserMapping.h

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

#pragma once

#include "loam_velodyne/Common.h"
#include "loam_velodyne/Twist.h"
#include "loam_velodyne/CircularBuffer.h"

#include "loam_velodyne/LaserMappingMetrics.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

namespace loam {

/* IMU state data. */
struct IMUState2
{
    /* The time of the measurement leading to this state (in seconds). */
    Time stamp;
    /* The current roll angle. */
    Angle roll;
    /* The current pitch angle. */
    Angle pitch;

    /** \brief Interpolate between two IMU states.
     *
     * @param start The first IMU state
     * @param end The second IMU state
     * @param ratio The interpolation ratio
     * @param result The target IMU state for storing the interpolation result
     */
    static void interpolate(const IMUState2& start, const IMUState2& end,
                            const float ratio, IMUState2& result)
    {
        const float invRatio = 1.0f - ratio;
        result.roll = start.roll.rad() * invRatio + end.roll.rad() * ratio;
        result.pitch = start.pitch.rad() * invRatio + end.pitch.rad() * ratio;
    }
};

class BasicLaserMapping
{
public:
    BasicLaserMapping(const float scanPeriod = 0.1f,
                      const std::size_t maxIterations = 10);

    /** \brief Try to process buffered data. */
    bool process(const Time& laserOdometryTime);
    void updateIMU(const IMUState2& newState);
    void updateOdometry(double pitch, double yaw, double roll,
                        double x, double y, double z);
    void updateOdometry(const Twist& twist);

    auto& laserCloud() { return *this->_laserCloudFullRes; }
    auto& laserCloudCornerLast() { return *this->_laserCloudCornerLast; }
    auto& laserCloudSurfLast() { return *this->_laserCloudSurfLast; }

    void setScanPeriod(float val) { this->_scanPeriod = val; }
    void setMaxIterations(size_t val) { this->_maxIterations = val; }
    void setDeltaTAbort(float val) { this->_deltaTAbort = val; }
    void setDeltaRAbort(float val) { this->_deltaRAbort = val; }

    auto& downSizeFilterCorner() { return this->_downSizeFilterCorner; }
    auto& downSizeFilterSurf() { return this->_downSizeFilterSurf; }
    auto& downSizeFilterMap() { return this->_downSizeFilterMap; }

    auto frameCount() const { return this->_frameCount; }
    auto scanPeriod() const { return this->_scanPeriod; }
    auto maxIterations() const { return this->_maxIterations; }
    auto deltaTAbort() const { return this->_deltaTAbort; }
    auto deltaRAbort() const { return this->_deltaRAbort; }

    const auto& transformAftMapped() const
    { return this->_transformAftMapped; }
    const auto& transformBefMapped() const
    { return this->_transformBefMapped; }
    const auto& laserCloudSurround() const
    { return *this->_laserCloudSurround; }
    const auto& laserCloudSurroundDS() const
    { return *this->_laserCloudSurroundDS; }

    inline bool hasFreshMap() const { return this->_downsizedMapCreated; }

private:
    /* Run an optimization. */
    void optimizeTransformTobeMapped();

    void transformAssociateToMap();
    void transformUpdate();

    // Transform the input point `pi` in the scan coordinate at t_(k + 2) to
    // the point `po` in the mapped coordinate frame, where t_(k + 2) is the
    // timestamp of the end of the sweep at t_(k + 1)
    pcl::PointXYZI pointAssociateToMap(const pcl::PointXYZI& pi);
    // Transform the input point `pi` in the mapped coordinate frame to the
    // point `po` in the scan coordinate at t_(k + 2), where t_(k + 2) is the
    // timestamp of the next scan (end time of the current scan)
    pcl::PointXYZI pointAssociateTobeMapped(const pcl::PointXYZI& pi);

    void transformFullResToMap();

    bool createDownsizedMap();

    // Compute a flattened index from the voxel index
    inline std::size_t toIndex(int i, int j, int k) const
    { return i + this->_laserCloudWidth * j
               + this->_laserCloudWidth * this->_laserCloudHeight * k; }

    // Check if the voxel is inside the map
    inline bool isVoxelInside(int i, int j, int k) const
    { return i >= 0 && i < this->_laserCloudWidth &&
             j >= 0 && j < this->_laserCloudHeight &&
             k >= 0 && k < this->_laserCloudDepth; }

    // Compute a voxel index from the mapped coordinate frame
    void toVoxelIndex(const float cubeSize, const float halfCubeSize,
                      const float tx, const float ty, const float tz,
                      int& idxI, int& idxJ, int& idxK) const;

    // Shift the map voxels along X axis
    void shiftMapVoxelsX(Eigen::Vector3i& centerCubeIdx);
    // Shift the map voxels along Y axis
    void shiftMapVoxelsY(Eigen::Vector3i& centerCubeIdx);
    // Shift the map voxels along Z axis
    void shiftMapVoxelsZ(Eigen::Vector3i& centerCubeIdx);

    // Check the occurrence of the degeneration
    bool checkDegeneration(const Eigen::Matrix<float, 6, 6>& hessianMat,
                           Eigen::Matrix<float, 6, 6>& projectionMat) const;
    // Compute the distances and coefficients from the point-to-edge
    // correspondences
    void computeCornerDistances();
    // Compute the distances and coefficients from the point-to-plane
    // correspondences
    void computePlaneDistances();

protected:
    // Clear the metrics message
    void clearMetricsMsg();

protected:
    // Flag to enable the metrics
    bool _metricsEnabled;
    // Metrics message
    loam_velodyne::LaserMappingMetrics _metricsMsg;

private:
    Time _laserOdometryTime;

    // Time between the consecutive scans
    float _scanPeriod;
    // Number of frames to skip
    const int _stackFrameNum;
    // Interval of the computation of the map of the surrounding
    const int _mapFrameNum;
    // Number of the frames processed
    long _frameCount;
    // Number of the frames processed since the last update of the map
    // of the surrounding
    long _mapFrameCount;

    // Maximum number of iterations
    std::size_t _maxIterations;
    // Optimization abort threshold for deltaT
    float _deltaTAbort;
    // Optimization abort threshold for deltaR
    float _deltaRAbort;

    // Coordinate of the voxel at the map center
    int _laserCloudCenWidth;
    // Coordinate of the voxel at the map center
    int _laserCloudCenHeight;
    // Coordinate of the voxel at the map center
    int _laserCloudCenDepth;
    // Width of the map in the number of voxels
    const std::size_t _laserCloudWidth;
    // Height of the map in the number of voxels
    const std::size_t _laserCloudHeight;
    // Depth of the map in the number of voxels
    const std::size_t _laserCloudDepth;
    // Number of the voxels in the map
    const std::size_t _laserCloudNum;

    // Last corner point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerLast;
    // Last surface point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfLast;
    // Last full resolution cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudFullRes;

    // Last corner point cloud projected to the mapped coordinate frame
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerStack;
    // Last surface point cloud projected to the mapped coordinate frame
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfStack;
    // Last corner point cloud downsampled for the pose optimization
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerStackDS;
    // Last surface point cloud downsampled for the pose optimization
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfStackDS;

    // Point cloud in the voxels in the field of view
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurround;
    // Point cloud in the voxels in the field of view downsampled for
    // computing the map of the surrounding
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurroundDS;
    // Corner point cloud in the voxels in the field of view used for
    // pose optimization
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudCornerFromMap;
    // Surface point cloud in the voxels in the field of view used for
    // pose optimization
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudSurfFromMap;

    // Point coordinates for pose optimization, where each coordinate is
    // in the local coordinate frame
    pcl::PointCloud<pcl::PointXYZI> _laserCloudOri;
    // Coefficients for pose optimization
    pcl::PointCloud<pcl::PointXYZI> _coeffSel;

    // Map with voxels for the corner point cloud around the current pose
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _laserCloudCornerArray;
    // Map with voxels for the surface point cloud around the current pose
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _laserCloudSurfArray;

    // Indices of the voxels in the field of view
    std::vector<std::size_t> _laserCloudValidInd;
    // Indices of the voxels around the current pose
    std::vector<std::size_t> _laserCloudSurroundInd;

    // Current odometry pose
    Twist _transformSum;
    // Update of the odometry pose
    Twist _transformIncre;
    // Current pose computed by the mapping
    Twist _transformTobeMapped;
    // Previous odometry pose
    Twist _transformBefMapped;
    // Previous pose computed by the mapping
    Twist _transformAftMapped;

    // History of IMU states
    CircularBuffer<IMUState2> _imuHistory;

    // Voxel filter for downsizing corner clouds
    pcl::VoxelGrid<pcl::PointXYZI> _downSizeFilterCorner;
    // Voxel filter for downsizing surface clouds
    pcl::VoxelGrid<pcl::PointXYZI> _downSizeFilterSurf;
    // Voxel filter for downsizing accumulated map
    pcl::VoxelGrid<pcl::PointXYZI> _downSizeFilterMap;

    bool _downsizedMapCreated = false;
};

} // namespace loam
