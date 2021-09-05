
// BasicLaserMapping.cpp

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

#include "loam_velodyne/BasicLaserMapping.h"
#include "loam_velodyne/Transform.hpp"
#include "loam_velodyne/nanoflann_pcl.h"
#include "math_utils.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace loam {

std::vector<Eigen::Vector3i> computeSurroundIndices()
{
    std::vector<Eigen::Vector3i> surroundIndices;
    surroundIndices.reserve(125);

    for (int i = -2; i <= 2; ++i)
        for (int j = -2; j <= 2; ++j)
            for (int k = -2; k <= 2; ++k)
                surroundIndices.emplace_back(i, j, k);

    return surroundIndices;
}

const std::vector<Eigen::Vector3i> kSurroundIndices = computeSurroundIndices();

BasicLaserMapping::BasicLaserMapping(const float scanPeriod,
                                     const std::size_t maxIterations) :
    _scanPeriod(scanPeriod),
    _stackFrameNum(1),
    _mapFrameNum(5),
    _frameCount(0),
    _mapFrameCount(0),
    _maxIterations(maxIterations),
    _deltaTAbort(0.05),
    _deltaRAbort(0.05),
    _laserCloudCenWidth(10),
    _laserCloudCenHeight(5),
    _laserCloudCenDepth(10),
    _laserCloudWidth(21),
    _laserCloudHeight(11),
    _laserCloudDepth(21),
    _laserCloudNum(_laserCloudWidth * _laserCloudHeight * _laserCloudDepth),
    _laserCloudCornerLast(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudSurfLast(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudFullRes(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudCornerStack(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudSurfStack(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudCornerStackDS(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudSurfStackDS(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudSurround(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudSurroundDS(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudCornerFromMap(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudSurfFromMap(new pcl::PointCloud<pcl::PointXYZI>())
{
    // Initialize frame counter
    this->_frameCount = this->_stackFrameNum - 1;
    this->_mapFrameCount = this->_mapFrameNum - 1;

    // Setup cloud vectors
    this->_laserCloudCornerArray.resize(this->_laserCloudNum);
    this->_laserCloudSurfArray.resize(this->_laserCloudNum);
    this->_laserCloudCornerDSArray.resize(this->_laserCloudNum);
    this->_laserCloudSurfDSArray.resize(this->_laserCloudNum);

    for (std::size_t i = 0; i < this->_laserCloudNum; ++i) {
        this->_laserCloudCornerArray[i].reset(
            new pcl::PointCloud<pcl::PointXYZI>());
        this->_laserCloudSurfArray[i].reset(
            new pcl::PointCloud<pcl::PointXYZI>());
        this->_laserCloudCornerDSArray[i].reset(
            new pcl::PointCloud<pcl::PointXYZI>());
        this->_laserCloudSurfDSArray[i].reset(
            new pcl::PointCloud<pcl::PointXYZI>());
    }

    // Setup down size filters
    this->_downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
    this->_downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
}

void BasicLaserMapping::transformAssociateToMap()
{
    // Compute three rotation matrices R_bc = Ry(bcy) Rx(bcx) Rz(bcz),
    // R_bl = Ry(bly) Rx(blx) Rz(blz), and R_al = Ry(aly) Rx(alx) Rz(alz)
    // and store three Euler angles (rx, ry, rz) that correspond to the
    // rotation matrix R_al (R_bl)^T R_bc, where (bcx, bcy, bcz) is
    // `_transformSum`, (blx, bly, blz) is `_transformBefMapped`, and
    // (alx, aly, alz) is `_transformAftMapped`

    // Create rotation matrices from Euler angles in `_transformSum`,
    // `_transformBefMapped`, and `_transformAftMapped`
    const Eigen::Matrix3f rotationMatSum = rotationMatrixZXY(
        this->_transformSum.rot_x.rad(),
        this->_transformSum.rot_y.rad(),
        this->_transformSum.rot_z.rad());
    const Eigen::Matrix3f rotationMatBefMapped = rotationMatrixZXY(
        this->_transformBefMapped.rot_x.rad(),
        this->_transformBefMapped.rot_y.rad(),
        this->_transformBefMapped.rot_z.rad());
    const Eigen::Matrix3f rotationMatAftMapped = rotationMatrixZXY(
        this->_transformAftMapped.rot_x.rad(),
        this->_transformAftMapped.rot_y.rad(),
        this->_transformAftMapped.rot_z.rad());

    // `_transformBefMapped` is actually `_transformSum` in the previous step,
    // meaning that it is the last odometry pose at t_(k + 1) computed using
    // the scan at t_k reprojected to t_(k + 1) and the scan at t_(k + 1)
    // `_transformSum` is the new odometry pose at t_(k + 2), and thus
    // `_transformIncre` is the difference of two consecutive odometry poses
    const Eigen::Vector3f globalIncre {
        this->_transformBefMapped.pos.x() - this->_transformSum.pos.x(),
        this->_transformBefMapped.pos.y() - this->_transformSum.pos.y(),
        this->_transformBefMapped.pos.z() - this->_transformSum.pos.z() };
    const Eigen::Vector3f transformIncre =
        rotationMatSum.transpose() * globalIncre;

    this->_transformIncre.pos.x() = transformIncre.x();
    this->_transformIncre.pos.y() = transformIncre.y();
    this->_transformIncre.pos.z() = transformIncre.z();

    // Compose three rotation matrices above for `_transformTobeMapped`
    // `_transformAftMapped` and `_transformTobeMapped` are the poses of the
    // scans at time t_(k + 1) and t_(k + 2) in the mapped coordinate frame
    const Eigen::Matrix3f rotationMatTobeMapped =
        rotationMatAftMapped * rotationMatBefMapped.transpose()
        * rotationMatSum;
    // Get three Euler angles from the rotation matrix above
    Eigen::Vector3f eulerAnglesTobeMapped;
    eulerAnglesFromRotationZXY(
        rotationMatTobeMapped, eulerAnglesTobeMapped.x(),
        eulerAnglesTobeMapped.y(), eulerAnglesTobeMapped.z());

    this->_transformTobeMapped.rot_x = eulerAnglesTobeMapped.x();
    this->_transformTobeMapped.rot_y = eulerAnglesTobeMapped.y();
    this->_transformTobeMapped.rot_z = eulerAnglesTobeMapped.z();

    // Compute the translation at t_(k + 2)
    const Eigen::Vector3f transformAftMapped {
        this->_transformAftMapped.pos.x(), this->_transformAftMapped.pos.y(),
        this->_transformAftMapped.pos.z() };
    const Eigen::Vector3f transformTobeMapped =
        transformAftMapped - rotationMatTobeMapped * transformIncre;

    this->_transformTobeMapped.pos.x() = transformTobeMapped.x();
    this->_transformTobeMapped.pos.y() = transformTobeMapped.y();
    this->_transformTobeMapped.pos.z() = transformTobeMapped.z();

    return;
}

void BasicLaserMapping::transformUpdate()
{
    auto timeDiff = [this](const std::size_t idx) {
        const auto imuTime = this->_imuHistory[idx].stamp;
        return toSec(this->_laserOdometryTime - imuTime) + this->_scanPeriod; };

    if (this->_imuHistory.size() > 0) {
        std::size_t imuIdx = 0;

        while (imuIdx < this->_imuHistory.size() - 1 && timeDiff(imuIdx) > 0)
            ++imuIdx;

        IMUState2 imuCur;

        if (imuIdx == 0 || timeDiff(imuIdx) > 0) {
            // Scan time newer then the newest or older than the oldest IMU data
            imuCur = this->_imuHistory[imuIdx];
        } else {
            const auto& imuState = this->_imuHistory[imuIdx];
            const auto& prevState = this->_imuHistory[imuIdx - 1];
            const float imuTimeDiff = toSec(imuState.stamp - prevState.stamp);
            const float ratio = -timeDiff(imuIdx) / imuTimeDiff;

            // Compute the IMU state at the current time by linear interpolation
            // (current time is `_laserOdometryTime` + `_scanPeriod`)
            IMUState2::interpolate(imuState, prevState, ratio, imuCur);
        }

        this->_transformTobeMapped.rot_x =
            0.998 * this->_transformTobeMapped.rot_x.rad()
            + 0.002 * imuCur.pitch.rad();
        this->_transformTobeMapped.rot_z =
            0.998 * this->_transformTobeMapped.rot_z.rad()
            + 0.002 * imuCur.roll.rad();
    }

    this->_transformBefMapped = this->_transformSum;
    this->_transformAftMapped = this->_transformTobeMapped;
}

void BasicLaserMapping::pointAssociateToMap(
    const pcl::PointXYZI& pi, pcl::PointXYZI& po)
{
    po.x = pi.x;
    po.y = pi.y;
    po.z = pi.z;
    po.intensity = pi.intensity;

    rotateZXY(po,
              this->_transformTobeMapped.rot_z,
              this->_transformTobeMapped.rot_x,
              this->_transformTobeMapped.rot_y);

    po.x += this->_transformTobeMapped.pos.x();
    po.y += this->_transformTobeMapped.pos.y();
    po.z += this->_transformTobeMapped.pos.z();
}

void BasicLaserMapping::pointAssociateTobeMapped(
    const pcl::PointXYZI& pi, pcl::PointXYZI& po)
{
    po.x = pi.x - this->_transformTobeMapped.pos.x();
    po.y = pi.y - this->_transformTobeMapped.pos.y();
    po.z = pi.z - this->_transformTobeMapped.pos.z();
    po.intensity = pi.intensity;

    rotateYXZ(po,
              -this->_transformTobeMapped.rot_y,
              -this->_transformTobeMapped.rot_x,
              -this->_transformTobeMapped.rot_z);
}

void BasicLaserMapping::transformFullResToMap()
{
    // Transform full resolution input cloud to map
    for (auto& pt : *this->_laserCloudFullRes)
        this->pointAssociateToMap(pt, pt);
}

bool BasicLaserMapping::createDownsizedMap()
{
    // Publish `/laser_cloud_surround` topic only when the surrounding point
    // cloud `_laserCloudSurround` is updated

    // Create new map cloud according to the input output ratio
    if (++this->_mapFrameCount < this->_mapFrameNum)
        return false;
    else
        this->_mapFrameCount = 0;

    // Accumulate map cloud
    this->_laserCloudSurround->clear();

    for (auto ind : _laserCloudSurroundInd) {
        *this->_laserCloudSurround += *this->_laserCloudCornerArray[ind];
        *this->_laserCloudSurround += *this->_laserCloudSurfArray[ind];
    }

    // Downsize map cloud
    this->_laserCloudSurroundDS->clear();
    this->_downSizeFilterCorner.setInputCloud(this->_laserCloudSurround);
    this->_downSizeFilterCorner.filter(*this->_laserCloudSurroundDS);

    return true;
}

/* Compute a voxel index from the mapped coordinate frame */
void BasicLaserMapping::toVoxelIndex(
    const float cubeSize, const float halfCubeSize,
    const float tx, const float ty, const float tz,
    int& idxI, int& idxJ, int& idxK) const
{
    const float cx = tx + halfCubeSize;
    const float cy = ty + halfCubeSize;
    const float cz = tz + halfCubeSize;

    idxI = static_cast<int>(cx / cubeSize) + this->_laserCloudCenWidth;
    idxJ = static_cast<int>(cy / cubeSize) + this->_laserCloudCenHeight;
    idxK = static_cast<int>(cz / cubeSize) + this->_laserCloudCenDepth;

    idxI = cx < 0.0f ? idxI - 1 : idxI;
    idxJ = cy < 0.0f ? idxJ - 1 : idxJ;
    idxK = cz < 0.0f ? idxK - 1 : idxK;
}

bool BasicLaserMapping::process(const Time& laserOdometryTime)
{
    if (++this->_frameCount < this->_stackFrameNum)
        return false;
    else
        this->_frameCount = 0;

    // `_laserOdometryTime` is actually t_(k + 1) in the paper (Section VI),
    // i.e., the timestamp of the current scan (or current sweep, since each
    // sweep contains only one scan in this implementation)
    this->_laserOdometryTime = laserOdometryTime;

    pcl::PointXYZI pointSel;

    // Relate incoming data to map
    // Compute the poses of the scans at t_(k + 1) and t_(k + 2) in the mapped
    // coordinate frame, i.e., `_transformAftMapped` and `_transformTobeMapped`
    this->transformAssociateToMap();

    // `_laserCloudCornerLast` and `_laserCloudSurfLast` are the sets of
    // corner and planar points in the scan at time t_(k + 1), and points in
    // `_laserCloudCornerLast` and `_laserCloudSurfLast` are reprojected to
    // the t_(k + 2), and `_transformTobeMapped` is used to transform the
    // coordinate at t_(k + 2) to the mapped coordinate frame

    for (const auto& pt : this->_laserCloudCornerLast->points) {
        this->pointAssociateToMap(pt, pointSel);
        this->_laserCloudCornerStack->push_back(pointSel);
    }

    for (const auto& pt : this->_laserCloudSurfLast->points) {
        this->pointAssociateToMap(pt, pointSel);
        this->_laserCloudSurfStack->push_back(pointSel);
    }

    pcl::PointXYZI pointOnYAxis;
    pointOnYAxis.x = 0.0f;
    pointOnYAxis.y = 10.0f;
    pointOnYAxis.z = 0.0f;
    this->pointAssociateToMap(pointOnYAxis, pointOnYAxis);

    // `CUBE_SIZE` and `CUBE_HALF` are in centimeters (50cm and 25cm)
    const auto CUBE_SIZE = 50.0f;
    const auto CUBE_HALF = CUBE_SIZE / 2.0f;

    // Compute the index of the center cube in the 10.5x5.5x10.5m cubic area,
    // and each cube stores the point cloud
    // `_laserCloudCenWidth`, `_laserCloudCenHeight`, and `_laserCloudCenDepth`
    // are initially set to 10, 5, and 10, respectively, and the number of
    // the cubes, i.e., `_laserCloudWidth`, `_laserCloudHeight`, and
    // `_laserCloudDepth` are initially set to 21, 11, and 21
    Eigen::Vector3i centerCubeIdx;
    this->toVoxelIndex(CUBE_SIZE, CUBE_HALF,
                       this->_transformTobeMapped.pos.x(),
                       this->_transformTobeMapped.pos.y(),
                       this->_transformTobeMapped.pos.z(),
                       centerCubeIdx.x(),
                       centerCubeIdx.y(),
                       centerCubeIdx.z());

    // Slide the cubes in `_laserCloudCornerArray` and `_laserCloudSurfArray`
    // along X axis to constrain the `centerCubeIdx.x()` to be within the range
    // of [3, `_laserCloudWidth` - 3)
    while (centerCubeIdx.x() < 3) {
        for (int j = 0; j < this->_laserCloudHeight; ++j) {
            for (int k = 0; k < this->_laserCloudDepth; ++k) {
                for (int i = this->_laserCloudWidth - 1; i >= 1; --i) {
                    const std::size_t indexA = this->toIndex(i, j, k);
                    const std::size_t indexB = this->toIndex(i - 1, j, k);
                    /* Only the pointers are swapped */
                    std::swap(this->_laserCloudCornerArray[indexA],
                              this->_laserCloudCornerArray[indexB]);
                    std::swap(this->_laserCloudSurfArray[indexA],
                              this->_laserCloudSurfArray[indexB]);
                }
                const std::size_t indexC = this->toIndex(0, j, k);
                this->_laserCloudCornerArray[indexC]->clear();
                this->_laserCloudSurfArray[indexC]->clear();
            }
        }
        ++centerCubeIdx.x();
        ++this->_laserCloudCenWidth;
    }

    while (centerCubeIdx.x() >= this->_laserCloudWidth - 3) {
        for (int j = 0; j < this->_laserCloudHeight; ++j) {
            for (int k = 0; k < this->_laserCloudDepth; ++k) {
                for (int i = 0; i < this->_laserCloudWidth - 1; ++i) {
                    const std::size_t indexA = this->toIndex(i, j, k);
                    const std::size_t indexB = this->toIndex(i + 1, j, k);
                    std::swap(this->_laserCloudCornerArray[indexA],
                              this->_laserCloudCornerArray[indexB]);
                    std::swap(this->_laserCloudSurfArray[indexA],
                              this->_laserCloudSurfArray[indexB]);
                }
                const std::size_t indexC =
                    this->toIndex(this->_laserCloudWidth - 1, j, k);
                this->_laserCloudCornerArray[indexC]->clear();
                this->_laserCloudSurfArray[indexC]->clear();
            }
        }
        --centerCubeIdx.x();
        --this->_laserCloudCenWidth;
    }

    // Slide the cubes in `_laserCloudCornerArray` and `_laserCloudSurfArray`
    // along Y axis to constrain the `centerCubeIdx.y()` to be within the range
    // of [3, `_laserCloudHeight` - 3)
    while (centerCubeIdx.y() < 3) {
        for (int i = 0; i < this->_laserCloudWidth; ++i) {
            for (int k = 0; k < this->_laserCloudDepth; ++k) {
                for (int j = this->_laserCloudHeight - 1; j >= 1; --j) {
                    const std::size_t indexA = this->toIndex(i, j, k);
                    const std::size_t indexB = this->toIndex(i, j - 1, k);
                    std::swap(this->_laserCloudCornerArray[indexA],
                              this->_laserCloudCornerArray[indexB]);
                    std::swap(this->_laserCloudSurfArray[indexA],
                              this->_laserCloudSurfArray[indexB]);
                }
                const std::size_t indexC = this->toIndex(i, 0, k);
                this->_laserCloudCornerArray[indexC]->clear();
                this->_laserCloudSurfArray[indexC]->clear();
            }
        }
        ++centerCubeIdx.y();
        ++this->_laserCloudCenHeight;
    }

    while (centerCubeIdx.y() >= this->_laserCloudHeight - 3) {
        for (int i = 0; i < this->_laserCloudWidth; ++i) {
            for (int k = 0; k < this->_laserCloudDepth; ++k) {
                for (int j = 0; j < this->_laserCloudHeight - 1; ++j) {
                    const std::size_t indexA = this->toIndex(i, j, k);
                    const std::size_t indexB = this->toIndex(i, j + 1, k);
                    std::swap(this->_laserCloudCornerArray[indexA],
                              this->_laserCloudCornerArray[indexB]);
                    std::swap(this->_laserCloudSurfArray[indexA],
                              this->_laserCloudSurfArray[indexB]);
                }
                const std::size_t indexC =
                    this->toIndex(i, this->_laserCloudHeight - 1, k);
                this->_laserCloudCornerArray[indexC]->clear();
                this->_laserCloudSurfArray[indexC]->clear();
            }
        }
        --centerCubeIdx.y();
        --this->_laserCloudCenHeight;
    }

    // Slide the cubes in `_laserCloudCornerArray` and `_laserCloudSurfArray`
    // along Z axis to constrain the `centerCubeIdx.z()` to be within the range
    // of [3, `_laserCloudDepth` - 3)
    while (centerCubeIdx.z() < 3) {
        for (int i = 0; i < this->_laserCloudWidth; ++i) {
            for (int j = 0; j < this->_laserCloudHeight; ++j) {
                for (int k = this->_laserCloudDepth - 1; k >= 1; --k) {
                    const std::size_t indexA = this->toIndex(i, j, k);
                    const std::size_t indexB = this->toIndex(i, j, k - 1);
                    std::swap(this->_laserCloudCornerArray[indexA],
                              this->_laserCloudCornerArray[indexB]);
                    std::swap(this->_laserCloudSurfArray[indexA],
                              this->_laserCloudSurfArray[indexB]);
                }
                const std::size_t indexC = this->toIndex(i, j, 0);
                this->_laserCloudCornerArray[indexC]->clear();
                this->_laserCloudSurfArray[indexC]->clear();
            }
        }
        ++centerCubeIdx.z();
        ++this->_laserCloudCenDepth;
    }

    while (centerCubeIdx.z() >= this->_laserCloudDepth - 3) {
        for (int i = 0; i < this->_laserCloudWidth; ++i) {
            for (int j = 0; j < this->_laserCloudHeight; ++j) {
                for (int k = 0; k < this->_laserCloudDepth - 1; ++k) {
                    const std::size_t indexA = this->toIndex(i, j, k);
                    const std::size_t indexB = this->toIndex(i, j, k + 1);
                    std::swap(this->_laserCloudCornerArray[indexA],
                              this->_laserCloudCornerArray[indexB]);
                    std::swap(this->_laserCloudSurfArray[indexA],
                              this->_laserCloudSurfArray[indexB]);
                }
                const std::size_t indexC =
                    this->toIndex(i, j, this->_laserCloudDepth - 1);
                this->_laserCloudCornerArray[indexC]->clear();
                this->_laserCloudSurfArray[indexC]->clear();
            }
        }
        --centerCubeIdx.z();
        --this->_laserCloudCenDepth;
    }

    // `_laserCloudValidInd` and `_laserCloudSurroundInd` contain 125 cube
    // indices at most when all cubes around the center cube (i, j, k)
    // are in the field of view, or all cubes have valid indices
    this->_laserCloudValidInd.clear();
    this->_laserCloudSurroundInd.clear();

    for (const auto& surroundIdx : kSurroundIndices) {
        const int i = centerCubeIdx.x() + surroundIdx.x();
        const int j = centerCubeIdx.y() + surroundIdx.y();
        const int k = centerCubeIdx.z() + surroundIdx.z();

        if (!this->isVoxelInside(i, j, k))
            continue;

        // Convert the voxel index to the mapped coordinate frame
        const float centerX = CUBE_SIZE * (i - this->_laserCloudCenWidth);
        const float centerY = CUBE_SIZE * (j - this->_laserCloudCenHeight);
        const float centerZ = CUBE_SIZE * (k - this->_laserCloudCenDepth);

        const pcl::PointXYZI transformPos =
            static_cast<pcl::PointXYZI>(this->_transformTobeMapped.pos);

        const int cornerOffsetX[8] = { -1, -1, -1, -1, 1, 1, 1, 1 };
        const int cornerOffsetY[8] = { -1, -1, 1, 1, -1, -1, 1, 1 };
        const int cornerOffsetZ[8] = { -1, 1, -1, 1, -1, 1, -1, 1 };

        // `corner` is the corner points of the cube at index (i, j, k)
        // in the mapped coordinate frame
        bool isInLaserFOV = false;

        for (int c = 0; c < 8; ++c) {
            pcl::PointXYZI corner;
            corner.x = centerX + CUBE_HALF * cornerOffsetX[c];
            corner.y = centerY + CUBE_HALF * cornerOffsetY[c];
            corner.z = centerZ + CUBE_HALF * cornerOffsetZ[c];

            const float squaredSide1 = calcSquaredDiff(transformPos, corner);
            const float squaredSide2 = calcSquaredDiff(pointOnYAxis, corner);

            // `100.0f + squaredSide1 - squaredSide2` equals to
            // `2 * 10 * sqrt(squaredSide1) * cos(x)` using law of
            // cosines, where `x` is `90 - (vertical angle)`
            const float check1 = pointOnYAxis.y * pointOnYAxis.y
                                 + squaredSide1 - squaredSide2;
            const float check2 = pointOnYAxis.y * std::sqrt(3.0f)
                                 * std::sqrt(squaredSide1);

            // This holds if |100.0f + side1 - side2| is less than
            // 10.0f * sqrt(3.0f) * sqrt(side1), which means that
            // the vertical angle of the point is within the range
            // of [-60, 60] (cos(x) is less than sqrt(3) / 2
            // and is larger than -sqrt(3) / 2, i.e., x is larger
            // than 30 degrees and is less than 150 degrees)
            if (std::abs(check1) < check2) {
                isInLaserFOV = true;
                break;
            }
        }

        const std::size_t cubeIdx = this->toIndex(i, j, k);

        if (isInLaserFOV)
            this->_laserCloudValidInd.push_back(cubeIdx);

        this->_laserCloudSurroundInd.push_back(cubeIdx);
    }

    // Prepare valid map corner and surface cloud for pose optimization
    this->_laserCloudCornerFromMap->clear();
    this->_laserCloudSurfFromMap->clear();

    for (const auto& ind : this->_laserCloudValidInd) {
        *this->_laserCloudCornerFromMap += *this->_laserCloudCornerArray[ind];
        *this->_laserCloudSurfFromMap += *this->_laserCloudSurfArray[ind];
    }

    // Prepare feature stack clouds for pose optimization
    // Convert the point coordinates from the mapped coordinate frame to the
    // scan coordinate frame at t_(k + 2)
    // After `pointAssociateTobeMapped()`, `_laserCloudCornerStack` is
    // basically the same as `_laserCloudCornerLast`
    for (auto& pt : *this->_laserCloudCornerStack)
        this->pointAssociateTobeMapped(pt, pt);

    // After `pointAssociateTobeMapped()`, `_laserCloudSurfStack` is
    // basically the same as `_laserCloudSurfLast`
    for (auto& pt : *this->_laserCloudSurfStack)
        this->pointAssociateTobeMapped(pt, pt);

    // Downsample feature stack clouds
    this->_laserCloudCornerStackDS->clear();
    this->_downSizeFilterCorner.setInputCloud(this->_laserCloudCornerStack);
    this->_downSizeFilterCorner.filter(*this->_laserCloudCornerStackDS);

    this->_laserCloudSurfStackDS->clear();
    this->_downSizeFilterSurf.setInputCloud(this->_laserCloudSurfStack);
    this->_downSizeFilterSurf.filter(*this->_laserCloudSurfStackDS);

    this->_laserCloudCornerStack->clear();
    this->_laserCloudSurfStack->clear();

    // Run pose optimization
    this->optimizeTransformTobeMapped();

    // Store downsized corner stack points in corresponding cube clouds
    for (int i = 0; i < this->_laserCloudCornerStackDS->size(); ++i) {
        // Convert the point coordinates from the scan frame to the map frame
        this->pointAssociateToMap(
            this->_laserCloudCornerStackDS->points[i], pointSel);

        // Compute the index of the cube corresponding to the point
        Eigen::Vector3i cubeIdx;
        this->toVoxelIndex(CUBE_SIZE, CUBE_HALF,
                           pointSel.x, pointSel.y, pointSel.z,
                           cubeIdx.x(), cubeIdx.y(), cubeIdx.z());

        // Append the aligned point to the cube
        if (!this->isVoxelInside(cubeIdx.x(), cubeIdx.y(), cubeIdx.z()))
            continue;

        const std::size_t cubeFlatIdx =
            this->toIndex(cubeIdx.x(), cubeIdx.y(), cubeIdx.z());
        this->_laserCloudCornerArray[cubeFlatIdx]->push_back(pointSel);
    }

    // Store downsized surface stack points in corresponding cube clouds
    for (int i = 0; i < this->_laserCloudSurfStackDS->size(); ++i) {
        this->pointAssociateToMap(
            this->_laserCloudSurfStackDS->points[i], pointSel);

        Eigen::Vector3i cubeIdx;
        this->toVoxelIndex(CUBE_SIZE, CUBE_HALF,
                           pointSel.x, pointSel.y, pointSel.z,
                           cubeIdx.x(), cubeIdx.y(), cubeIdx.z());

        if (!this->isVoxelInside(cubeIdx.x(), cubeIdx.y(), cubeIdx.z()))
            continue;

        const std::size_t cubeFlatIdx =
            this->toIndex(cubeIdx.x(), cubeIdx.y(), cubeIdx.z());
        this->_laserCloudSurfArray[cubeFlatIdx]->push_back(pointSel);
    }

    // Downsize all valid (within field of view) feature cube clouds
    // TODO: `_laserCloudCornerDSArray` and `_laserCloudSurfDSArray` are
    // not necessary
    for (const auto& ind : this->_laserCloudValidInd) {
        this->_laserCloudCornerDSArray[ind]->clear();
        this->_downSizeFilterCorner.setInputCloud(
            this->_laserCloudCornerArray[ind]);
        this->_downSizeFilterCorner.filter(
            *this->_laserCloudCornerDSArray[ind]);

        this->_laserCloudSurfDSArray[ind]->clear();
        this->_downSizeFilterSurf.setInputCloud(
            this->_laserCloudSurfArray[ind]);
        this->_downSizeFilterSurf.filter(
            *this->_laserCloudSurfDSArray[ind]);

        // Swap cube clouds for next processing
        this->_laserCloudCornerArray[ind].swap(
            this->_laserCloudCornerDSArray[ind]);
        this->_laserCloudSurfArray[ind].swap(
            this->_laserCloudSurfDSArray[ind]);
    }

    this->transformFullResToMap();
    this->_downsizedMapCreated = this->createDownsizedMap();

    return true;
}

void BasicLaserMapping::updateIMU(const IMUState2& newState)
{
    this->_imuHistory.push(newState);
}

void BasicLaserMapping::updateOdometry(double pitch, double yaw, double roll,
                                       double x, double y, double z)
{
    this->_transformSum.rot_x = pitch;
    this->_transformSum.rot_y = yaw;
    this->_transformSum.rot_z = roll;

    this->_transformSum.pos.x() = static_cast<float>(x);
    this->_transformSum.pos.y() = static_cast<float>(y);
    this->_transformSum.pos.z() = static_cast<float>(z);
}

void BasicLaserMapping::updateOdometry(const Twist& twist)
{
    this->_transformSum = twist;
}

nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerFromMap;
nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfFromMap;

void BasicLaserMapping::optimizeTransformTobeMapped()
{
    if (this->_laserCloudCornerFromMap->size() <= 10 ||
        this->_laserCloudSurfFromMap->size() <= 100)
        return;

    kdtreeCornerFromMap.setInputCloud(this->_laserCloudCornerFromMap);
    kdtreeSurfFromMap.setInputCloud(this->_laserCloudSurfFromMap);

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    // Start the iterations of the Gauss-Newton method
    for (std::size_t iter = 0; iter < this->_maxIterations; ++iter) {
        this->_laserCloudOri.clear();
        this->_coeffSel.clear();

        // Compute the distances and coefficients from the point-to-edge
        // correspondences
        this->computeCornerDistances();
        // Compute the distances and coefficients from the point-to-plane
        // correspondences
        this->computePlaneDistances();

        const std::size_t laserCloudSelNum = this->_laserCloudOri.size();

        if (laserCloudSelNum < 50)
            continue;

        const float rotX = this->_transformTobeMapped.rot_x.rad();
        const float rotY = this->_transformTobeMapped.rot_y.rad();
        const float rotZ = this->_transformTobeMapped.rot_z.rad();

        // Create a rotation matrix from `_transformTobeMapped`
        const Eigen::Matrix3f rotationMatTobeMapped =
            rotationMatrixZXY(rotX, rotY, rotZ);
        // Compute partial derivatives of the rotation matrix above
        const Eigen::Matrix3f rotationMatParX =
            partialXFromRotationZXY(rotX, rotY, rotZ);
        const Eigen::Matrix3f rotationMatParY =
            partialYFromRotationZXY(rotX, rotY, rotZ);
        const Eigen::Matrix3f rotationMatParZ =
            partialZFromRotationZXY(rotX, rotY, rotZ);

        // `matA` is the Jacobian matrix in Equation (12)
        Eigen::Matrix<float, Eigen::Dynamic, 6> matA(laserCloudSelNum, 6);
        Eigen::Matrix<float, 6, Eigen::Dynamic> matAt(6, laserCloudSelNum);
        // `matB` is the distance vector (d) in Equation (12)
        Eigen::VectorXf vecB(laserCloudSelNum);

        for (int i = 0; i < laserCloudSelNum; ++i) {
            const pcl::PointXYZI& pointOri = this->_laserCloudOri.points[i];
            const pcl::PointXYZI& coeff = this->_coeffSel.points[i];

            // Compute a partial derivative of the point-to-edge or
            // point-to-plane distance with respect to the rotation
            pcl::Vector3fMapConst vecPoint = pointOri.getVector3fMap();
            pcl::Vector3fMapConst vecCoeff = coeff.getVector3fMap();
            const Eigen::Vector3f vecGradRot {
                (rotationMatParX * vecPoint).transpose() * vecCoeff,
                (rotationMatParY * vecPoint).transpose() * vecCoeff,
                (rotationMatParZ * vecPoint).transpose() * vecCoeff };
            matA.block<1, 3>(i, 0) = vecGradRot;

            // Partial derivative of the point-to-edge or point-to-plane
            // distance with respect to the translation equals to the
            // coefficient, i.e., the normal vector, which is already obtained
            matA.block<1, 3>(i, 3) = vecCoeff;

            // Point-to-edge or point-to-plane distance is stored in the
            // intensity field in the coefficient
            // Reverse the sign of the residual to follow Gauss-Newton method
            vecB(i, 0) = -coeff.intensity;
        }

        matAt = matA.transpose();
        // `matAtA` is the Hessian matrix (J^T J) in Equation (12)
        const Eigen::Matrix<float, 6, 6> matAtA = matAt * matA;
        // `matAtB` is the residual vector (J^T d) in Equation (12)
        const Eigen::VectorXf vecAtB = matAt * vecB;
        // Compute the increment to the current transformation
        Eigen::Matrix<float, 6, 1> vecX =
            matAtA.colPivHouseholderQr().solve(vecAtB);

        // Check the occurrence of the degeneration
        if (iter == 0)
            isDegenerate = this->checkDegeneration(matAtA, matP);

        // Do not update the transformation along the degenerate direction
        if (isDegenerate) {
            Eigen::Matrix<float, 6, 1> vecX2(vecX);
            vecX = matP * vecX2;
        }

        // Update the transformation (rotation and translation)
        this->_transformTobeMapped.rot_x += vecX(0, 0);
        this->_transformTobeMapped.rot_y += vecX(1, 0);
        this->_transformTobeMapped.rot_z += vecX(2, 0);
        this->_transformTobeMapped.pos.x() += vecX(3, 0);
        this->_transformTobeMapped.pos.y() += vecX(4, 0);
        this->_transformTobeMapped.pos.z() += vecX(5, 0);

        // Compute the increment in degrees and centimeters
        const float deltaR = std::sqrt(
            std::pow(rad2deg(vecX(0, 0)), 2)
            + std::pow(rad2deg(vecX(1, 0)), 2)
            + std::pow(rad2deg(vecX(2, 0)), 2));
        const float deltaT = std::sqrt(
            std::pow(vecX(3, 0) * 100, 2)
            + std::pow(vecX(4, 0) * 100, 2)
            + std::pow(vecX(5, 0) * 100, 2));

        // Terminate the Gauss-Newton method if the increment is small
        if (deltaR < this->_deltaRAbort && deltaT < this->_deltaTAbort)
            break;
    }

    // Refine the transformation using IMU data and update the transformation
    this->transformUpdate();
}

// Check the occurrence of the degeneration
bool BasicLaserMapping::checkDegeneration(
    const Eigen::Matrix<float, 6, 6>& hessianMat,
    Eigen::Matrix<float, 6, 6>& projectionMat) const
{
    // Check the occurrence of the degeneration following the paper:
    // Ji Zhang, Michael Kaess, and Sanjiv Singh. "On Degeneracy of
    // Optimization-based State Estimation Problems," in the Proceedings of the
    // IEEE International Conference on Robotics and Automation (ICRA), 2016.

    // Compute the eigenvalues and eigenvectors of the Hessian matrix
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eigenSolver;
    eigenSolver.compute(hessianMat);

    const Eigen::Matrix<float, 1, 6> matE = eigenSolver.eigenvalues().real();
    const Eigen::Matrix<float, 6, 6> matV = eigenSolver.eigenvectors().real();
    Eigen::Matrix<float, 6, 6> matV2 = matV;

    bool isDegenerate = false;
    const float eigenThreshold[6] = { 100, 100, 100, 100, 100, 100 };

    // Eigenvalues are sorted in the increasing order
    // Detect the occurrence of the degeneration if one of the eigenvalues is
    // less than 100
    for (int i = 0; i < 6; ++i) {
        if (matE(0, i) < eigenThreshold[i]) {
            matV2.row(i).setZero();
            isDegenerate = true;
        } else {
            break;
        }
    }

    // Do not update the transformation along the degenerate direction
    projectionMat = matV.inverse() * matV2;

    return isDegenerate;
}

// Compute the distances and coefficients from the point-to-edge
// correspondences
void BasicLaserMapping::computeCornerDistances()
{
    const std::size_t laserCloudCornerStackNum =
        this->_laserCloudCornerStackDS->size();

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    pointSearchInd.resize(5, 0);
    pointSearchSqDis.resize(5, 0.0f);

    // For each corner point in the downsampled current point cloud,
    // find the closest neighbors in the map cloud
    for (int i = 0; i < laserCloudCornerStackNum; ++i) {
        const pcl::PointXYZI pointOri =
            this->_laserCloudCornerStackDS->points[i];

        // Convert the corner point coordinates in the scan coordinate frame
        // at t_(k + 2) to the mapped coordinate frame using the current pose
        // estimate, i.e., `_transformTobeMapped`
        pcl::PointXYZI pointSel;
        this->pointAssociateToMap(pointOri, pointSel);
        // Find the 5 closest neighbors in the map cloud
        kdtreeCornerFromMap.nearestKSearch(
            pointSel, 5, pointSearchInd, pointSearchSqDis);

        // If distances to all closest neighbors are less than 1m, then
        // compute the coefficient for the Gauss-Newton optimization
        if (pointSearchSqDis[4] >= 1.0f)
            continue;

        // Store the average of the closest neighbor coordinates
        Eigen::Matrix<float, 5, 3> matClosestPoints;

        for (int j = 0; j < 5; ++j) {
            const pcl::PointXYZI& closestPoint =
                this->_laserCloudCornerFromMap->at(pointSearchInd[j]);
            matClosestPoints.row(j) = closestPoint.getVector3fMap();
        }

        // Compute the average of the closest neighbor coordinates
        const Eigen::Vector3f vecMean = matClosestPoints.colwise().mean();
        // Compute the lower-triangular part of the covariance matrix
        // of the closest neighbor coordinates
        const Eigen::Matrix<float, 5, 3> matCentered =
            matClosestPoints.rowwise() - vecMean.transpose();
        Eigen::Matrix3f matCovariance;
        matCovariance.triangularView<Eigen::Lower>() =
            matCentered.transpose() * matCentered / 5.0f;

        // Compute eigenvalues and eigenvectors of the covariance matrix
        // This solver only looks at the lower-triangular part
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver;
        eigenSolver.compute(matCovariance);
        // Eigenvalues are sorted in an ascending order
        const Eigen::Vector3f vecD1 = eigenSolver.eigenvalues().real();
        const Eigen::Matrix3f matV1 = eigenSolver.eigenvectors().real();

        // If one eigenvalue is larger than the other two, then the closest
        // neighbors represent the edge line, and the eigenvector associated
        // with the largest eigenvalue represents the orientation of the
        // edge line
        if (vecD1[2] <= 3.0f * vecD1[1])
            continue;

        // Extract the orientation of the edge line
        const Eigen::Vector3f vecEdgeLine = matV1.col(2);

        const Eigen::Vector3f vecI = pointSel.getVector3fMap();
        const Eigen::Vector3f vecJ = vecMean + 0.1f * vecEdgeLine;
        const Eigen::Vector3f vecL = vecMean - 0.1f * vecEdgeLine;

        const Eigen::Vector3f vecIJ = vecI - vecJ;
        const Eigen::Vector3f vecIL = vecI - vecL;
        const Eigen::Vector3f vecJL = vecJ - vecL;
        const Eigen::Vector3f vecCross = vecIJ.cross(vecIL);

        // Compute the numerator of the Equation (2)
        const float a012 = vecCross.norm();
        // Compute the denominator of the Equation (2)
        const float l12 = vecJL.norm();

        // Compute the normal vector (la, lb, lc) of the edge line, i.e.,
        // the vector from the projection of the point (x0, y0, z0) on
        // the edge line between points (x1, y1, z1) and (x2, y2, z2)
        // to the point (x0, y0, z0)
        const Eigen::Vector3f vecNormal = vecJL.cross(vecCross) / a012 / l12;
        // Compute the point-to-line distance using the Equation (2),
        // i.e., the distance from the corner point in the current scan
        // (x0, y0, z0) to the edge lines between (x1, y1, z1) and
        // (x2, y2, z2) which are obtained from the corner points of
        // the map cloud
        const float ld2 = a012 / l12;

        // Compute the bisquare weight
        const float s = 1.0f - 0.9f * std::fabs(ld2);

        if (s <= 0.1f)
            continue;

        // Compute the coefficient for the pose optimization
        pcl::PointXYZI coeff;
        coeff.getVector3fMap() = s * vecNormal;
        coeff.intensity = s * ld2;

        this->_laserCloudOri.push_back(pointOri);
        this->_coeffSel.push_back(coeff);
    }
}

// Compute the distances and coefficients from the point-to-plane
// correspondences
void BasicLaserMapping::computePlaneDistances()
{
    const std::size_t laserCloudSurfStackNum =
        this->_laserCloudSurfStackDS->size();

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    pointSearchInd.resize(5, 0);
    pointSearchSqDis.resize(5, 0.0f);

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    matA0.setZero();
    matB0.setConstant(-1.0f);

    // For each planar point in the downsampled current point cloud,
    // find the closest neighbors in the map cloud
    for (int i = 0; i < laserCloudSurfStackNum; ++i) {
        const pcl::PointXYZI pointOri = _laserCloudSurfStackDS->points[i];
        // Convert the planar point coordinates in the scan coordinate frame
        // at t_(k + 2) to the mapped coordinate frame using the current pose
        // estimate
        pcl::PointXYZI pointSel;
        this->pointAssociateToMap(pointOri, pointSel);
        // Find the 5 closest neighbors in the map cloud
        kdtreeSurfFromMap.nearestKSearch(
            pointSel, 5, pointSearchInd, pointSearchSqDis);

        // If distances to all closest neighbors are less than 1m, then
        // compute the coefficient for the Gauss-Newton optimization
        if (pointSearchSqDis[4] >= 1.0f)
            continue;

        // Store coordinates of the closest neighbors to the matrix
        for (int j = 0; j < 5; ++j) {
            const pcl::PointXYZI& neighborPoint =
                this->_laserCloudSurfFromMap->at(pointSearchInd[j]);
            matA0.row(j) = neighborPoint.getVector3fMap();
        }

        // Compute the normal vector (pa, pb, pc) that is perpenidcular to
        // the plane defined by a set of neighbor points `pointSearchInd`
        const Eigen::Vector3f vecX0 = matA0.colPivHouseholderQr().solve(matB0);

        // Normalize the normal vector (pa, pb, pc) of the plane
        const float ps = vecX0.norm();
        const float pd = 1.0f / ps;
        const Eigen::Vector3f vecNormal = vecX0 / ps;

        // If all neighbor points `pointSearchInd` are on the same plane,
        // then the distance between the neighbor point and the plane
        // should be closer to zero
        const Eigen::Matrix<float, 1, 5> pointToPlaneDists =
            (vecNormal.transpose() * matA0.transpose()).array() + pd;

        // If a distance between a neighbor point and the plane is larger
        // than 0.2, then the neighbor points are not on the same plane
        if ((pointToPlaneDists.cwiseAbs().array() > 0.2f).any())
            continue;

        // Compute the d_h using the Equation (3)
        // Note that the distance below could be negative
        const float pd2 = vecNormal.transpose()
                          * pointSel.getVector3fMap() + pd;

        // Compute the bisquare weight
        const float s = 1.0f - 0.9f * std::fabs(pd2)
                        / std::sqrt(calcPointDistance(pointSel));

        if (s <= 0.1f)
            continue;

        pcl::PointXYZI coeff;
        coeff.getVector3fMap() = s * vecNormal;
        coeff.intensity = s * pd2;

        this->_laserCloudOri.push_back(pointOri);
        this->_coeffSel.push_back(coeff);
    }
}

} // namespace loam
