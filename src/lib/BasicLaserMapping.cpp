
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
#include "loam_velodyne/nanoflann_pcl.h"
#include "math_utils.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace loam {

using std::sqrt;
using std::fabs;
using std::asin;
using std::atan2;
using std::pow;

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
   // `_transformBefMapped` is actually `_transformSum` in the previous step,
   // meaning that it is the last odometry pose at t_(k + 1) computed using
   // the scan at t_k reprojected to t_(k + 1) and the scan at t_(k + 1)
   // `_transformSum` is the new odometry pose at t_(k + 2), and thus
   // `_transformIncre` is the difference of two consecutive odometry poses
   _transformIncre.pos = _transformBefMapped.pos - _transformSum.pos;
   rotateYXZ(_transformIncre.pos, -(_transformSum.rot_y), -(_transformSum.rot_x), -(_transformSum.rot_z));

   // `_transformAftMapped` and `_transformTobeMapped` are the poses of the
   // scans at time t_(k + 1) and t_(k + 2) in the mapped coordinate frame

   // Compute three rotation matrices R_bc = Ry(bcy) Rx(bcx) Rz(bcz),
   // R_bl = Ry(bly) Rx(blx) Rz(blz), and R_al = Ry(aly) Rx(alx) Rz(alz)
   // and store three Euler angles (rx, ry, rz) that correspond to the
   // rotation matrix R_al (R_bl)^T R_bc, where (bcx, bcy, bcz) is
   // `_transformSum`, (blx, bly, blz) is `_transformBefMapped`, and
   // (alx, aly, alz) is `_transformAftMapped`

   float sbcx = _transformSum.rot_x.sin();
   float cbcx = _transformSum.rot_x.cos();
   float sbcy = _transformSum.rot_y.sin();
   float cbcy = _transformSum.rot_y.cos();
   float sbcz = _transformSum.rot_z.sin();
   float cbcz = _transformSum.rot_z.cos();

   float sblx = _transformBefMapped.rot_x.sin();
   float cblx = _transformBefMapped.rot_x.cos();
   float sbly = _transformBefMapped.rot_y.sin();
   float cbly = _transformBefMapped.rot_y.cos();
   float sblz = _transformBefMapped.rot_z.sin();
   float cblz = _transformBefMapped.rot_z.cos();

   float salx = _transformAftMapped.rot_x.sin();
   float calx = _transformAftMapped.rot_x.cos();
   float saly = _transformAftMapped.rot_y.sin();
   float caly = _transformAftMapped.rot_y.cos();
   float salz = _transformAftMapped.rot_z.sin();
   float calz = _transformAftMapped.rot_z.cos();

   float srx = -sbcx * (salx*sblx + calx * cblx*salz*sblz + calx * calz*cblx*cblz)
      - cbcx * sbcy*(calx*calz*(cbly*sblz - cblz * sblx*sbly)
                     - calx * salz*(cbly*cblz + sblx * sbly*sblz) + cblx * salx*sbly)
      - cbcx * cbcy*(calx*salz*(cblz*sbly - cbly * sblx*sblz)
                     - calx * calz*(sbly*sblz + cbly * cblz*sblx) + cblx * cbly*salx);
   _transformTobeMapped.rot_x = -asin(srx);

   float srycrx = sbcx * (cblx*cblz*(caly*salz - calz * salx*saly)
                          - cblx * sblz*(caly*calz + salx * saly*salz) + calx * saly*sblx)
      - cbcx * cbcy*((caly*calz + salx * saly*salz)*(cblz*sbly - cbly * sblx*sblz)
                     + (caly*salz - calz * salx*saly)*(sbly*sblz + cbly * cblz*sblx) - calx * cblx*cbly*saly)
      + cbcx * sbcy*((caly*calz + salx * saly*salz)*(cbly*cblz + sblx * sbly*sblz)
                     + (caly*salz - calz * salx*saly)*(cbly*sblz - cblz * sblx*sbly) + calx * cblx*saly*sbly);
   float crycrx = sbcx * (cblx*sblz*(calz*saly - caly * salx*salz)
                          - cblx * cblz*(saly*salz + caly * calz*salx) + calx * caly*sblx)
      + cbcx * cbcy*((saly*salz + caly * calz*salx)*(sbly*sblz + cbly * cblz*sblx)
                     + (calz*saly - caly * salx*salz)*(cblz*sbly - cbly * sblx*sblz) + calx * caly*cblx*cbly)
      - cbcx * sbcy*((saly*salz + caly * calz*salx)*(cbly*sblz - cblz * sblx*sbly)
                     + (calz*saly - caly * salx*salz)*(cbly*cblz + sblx * sbly*sblz) - calx * caly*cblx*sbly);
   _transformTobeMapped.rot_y = atan2(srycrx / _transformTobeMapped.rot_x.cos(),
                                      crycrx / _transformTobeMapped.rot_x.cos());

   float srzcrx = (cbcz*sbcy - cbcy * sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly * sblx*sblz)
                                                  - calx * calz*(sbly*sblz + cbly * cblz*sblx) + cblx * cbly*salx)
      - (cbcy*cbcz + sbcx * sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz * sblx*sbly)
                                        - calx * salz*(cbly*cblz + sblx * sbly*sblz) + cblx * salx*sbly)
      + cbcx * sbcz*(salx*sblx + calx * cblx*salz*sblz + calx * calz*cblx*cblz);
   float crzcrx = (cbcy*sbcz - cbcz * sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz * sblx*sbly)
                                                  - calx * salz*(cbly*cblz + sblx * sbly*sblz) + cblx * salx*sbly)
      - (sbcy*sbcz + cbcy * cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly * sblx*sblz)
                                        - calx * calz*(sbly*sblz + cbly * cblz*sblx) + cblx * cbly*salx)
      + cbcx * cbcz*(salx*sblx + calx * cblx*salz*sblz + calx * calz*cblx*cblz);
   _transformTobeMapped.rot_z = atan2(srzcrx / _transformTobeMapped.rot_x.cos(),
                                      crzcrx / _transformTobeMapped.rot_x.cos());

   Vector3 v = _transformIncre.pos;
   rotateZXY(v, _transformTobeMapped.rot_z, _transformTobeMapped.rot_x, _transformTobeMapped.rot_y);
   _transformTobeMapped.pos = _transformAftMapped.pos - v;
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

    Eigen::Matrix3f matV1;
    matV1.setZero();

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

        const float srx = this->_transformTobeMapped.rot_x.sin();
        const float crx = this->_transformTobeMapped.rot_x.cos();
        const float sry = this->_transformTobeMapped.rot_y.sin();
        const float cry = this->_transformTobeMapped.rot_y.cos();
        const float srz = this->_transformTobeMapped.rot_z.sin();
        const float crz = this->_transformTobeMapped.rot_z.cos();

        const std::size_t laserCloudSelNum = this->_laserCloudOri.size();

        if (laserCloudSelNum < 50)
            continue;

        // `matA` is the Jacobian matrix in Equation (12)
        Eigen::Matrix<float, Eigen::Dynamic, 6> matA(laserCloudSelNum, 6);
        Eigen::Matrix<float, 6, Eigen::Dynamic> matAt(6, laserCloudSelNum);
        // `matAtA` is the Hessian matrix (J^T J) in Equation (12)
        Eigen::Matrix<float, 6, 6> matAtA;
        // `matB` is the distance vector (d) in Equation (12)
        Eigen::VectorXf matB(laserCloudSelNum);
        // `matAtB` is the residual vector (J^T d) in Equation (12)
        Eigen::VectorXf matAtB;
        Eigen::VectorXf matX;

        for (int i = 0; i < laserCloudSelNum; ++i) {
            const pcl::PointXYZI& pointOri = this->_laserCloudOri.points[i];
            const pcl::PointXYZI& coeff = this->_coeffSel.points[i];

            const float arx = (crx*sry*srz*pointOri.x + crx * crz*sry*pointOri.y - srx * sry*pointOri.z) * coeff.x
                + (-srx * srz*pointOri.x - crz * srx*pointOri.y - crx * pointOri.z) * coeff.y
                + (crx*cry*srz*pointOri.x + crx * cry*crz*pointOri.y - cry * srx*pointOri.z) * coeff.z;

            const float ary = ((cry*srx*srz - crz * sry)*pointOri.x
                        + (sry*srz + cry * crz*srx)*pointOri.y + crx * cry*pointOri.z) * coeff.x
                + ((-cry * crz - srx * sry*srz)*pointOri.x
                + (cry*srz - crz * srx*sry)*pointOri.y - crx * sry*pointOri.z) * coeff.z;

            const float arz = ((crz*srx*sry - cry * srz)*pointOri.x + (-cry * crz - srx * sry*srz)*pointOri.y)*coeff.x
                + (crx*crz*pointOri.x - crx * srz*pointOri.y) * coeff.y
                + ((sry*srz + cry * crz*srx)*pointOri.x + (crz*sry - cry * srx*srz)*pointOri.y)*coeff.z;

            matA(i, 0) = arx;
            matA(i, 1) = ary;
            matA(i, 2) = arz;
            matA(i, 3) = coeff.x;
            matA(i, 4) = coeff.y;
            matA(i, 5) = coeff.z;
            // Reverse the sign of the residual to follow Gauss-Newton method
            matB(i, 0) = -coeff.intensity;
        }

        matAt = matA.transpose();
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        // Compute the increment to the current transformation
        matX = matAtA.colPivHouseholderQr().solve(matAtB);

        // Check the occurrence of the degeneration following the paper:
        // Ji Zhang, Michael Kaess, and Sanjiv Singh. "On Degeneracy of
        // Optimization-based State Estimation Problems," in the Proceedings
        // of the IEEE International Conference on Robotics and Automation
        // (ICRA), 2016.
        if (iter == 0) {
            // Compute the eigenvalues and eigenvectors of the Hessian matrix
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>>
                eigenSolver(matAtA);
            const Eigen::Matrix<float, 1, 6> matE =
                eigenSolver.eigenvalues().real();
            const Eigen::Matrix<float, 6, 6> matV =
                eigenSolver.eigenvectors().real();
            Eigen::Matrix<float, 6, 6> matV2 = matV;

            isDegenerate = false;
            const float eigenThreshold[6] = { 100, 100, 100, 100, 100, 100 };

            // Eigenvalues are sorted in the increasing order
            // Detect the occurrence of the degeneration if one of the
            // eigenvalues is less than 100
            for (int i = 0; i < 6; ++i) {
                if (matE(0, i) < eigenThreshold[i]) {
                    matV2.row(i).setZero();
                    isDegenerate = true;
                } else {
                    break;
                }
            }

            // Do not update the transformation along the degenerate direction
            matP = matV.inverse() * matV2;
        }

        if (isDegenerate) {
            // Do not update the transformation along the degenerate direction
            Eigen::Matrix<float, 6, 1> matX2(matX);
            matX = matP * matX2;
        }

        // Update the transformation (rotation and translation)
        this->_transformTobeMapped.rot_x += matX(0, 0);
        this->_transformTobeMapped.rot_y += matX(1, 0);
        this->_transformTobeMapped.rot_z += matX(2, 0);
        this->_transformTobeMapped.pos.x() += matX(3, 0);
        this->_transformTobeMapped.pos.y() += matX(4, 0);
        this->_transformTobeMapped.pos.z() += matX(5, 0);

        // Compute the increment in degrees and centimeters
        const float deltaR = std::sqrt(
            std::pow(rad2deg(matX(0, 0)), 2)
            + std::pow(rad2deg(matX(1, 0)), 2)
            + std::pow(rad2deg(matX(2, 0)), 2));
        const float deltaT = std::sqrt(
            std::pow(matX(3, 0) * 100, 2)
            + std::pow(matX(4, 0) * 100, 2)
            + std::pow(matX(5, 0) * 100, 2));

        // Terminate the Gauss-Newton method if the increment is small
        if (deltaR < this->_deltaRAbort && deltaT < this->_deltaTAbort)
            break;
    }

    // Refine the transformation using IMU data and update the transformation
    this->transformUpdate();
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

        // Compute the average of the closest neighbor coordinates
        Vector3 vc { 0.0f, 0.0f, 0.0f };

        for (const auto idx : pointSearchInd)
            vc += Vector3(this->_laserCloudCornerFromMap->at(idx));

        vc /= 5.0f;

        // Compute the lower-triangular part of the covariance matrix of
        // the closest neighbor coordinates and then compute eigenvectors
        // and eigenvalues of the covariance matrix
        Eigen::Matrix3f matCov = Eigen::Matrix3f::Zero();

        for (const auto idx : pointSearchInd) {
            const Vector3 vecDiff =
                Vector3(this->_laserCloudCornerFromMap->at(idx)) - vc;

            matCov(0, 0) += vecDiff.x() * vecDiff.x();
            matCov(1, 0) += vecDiff.x() * vecDiff.y();
            matCov(2, 0) += vecDiff.x() * vecDiff.z();
            matCov(1, 1) += vecDiff.y() * vecDiff.y();
            matCov(2, 1) += vecDiff.y() * vecDiff.z();
            matCov(2, 2) += vecDiff.z() * vecDiff.z();
        }

        const Eigen::Matrix3f matA1 = matCov / 5.0f;

        // This solver only looks at the lower-triangular part of matA1.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA1);
        // Eigenvalues are sorted in an ascending order
        const Eigen::Vector3f vecD1 = eigenSolver.eigenvalues().real();
        const Eigen::Matrix3f matV1 = eigenSolver.eigenvectors().real();

        // If one eigenvalue is larger than the other two, then the closest
        // neighbors represent the edge line, and the eigenvector associated
        // with the largest eigenvalue represents the orientation of the
        // edge line
        if (vecD1[2] <= 3.0f * vecD1[1])
            continue;

        const float x0 = pointSel.x;
        const float y0 = pointSel.y;
        const float z0 = pointSel.z;
        // The position of the edge line is the center of the 5 closest
        // neighbors that represent the edge line, and two points
        // (x1, y1, z1) and (x2, y2, z2) should be on the edge line
        const float x1 = vc.x() + 0.1f * matV1(0, 2);
        const float y1 = vc.y() + 0.1f * matV1(1, 2);
        const float z1 = vc.z() + 0.1f * matV1(2, 2);
        const float x2 = vc.x() - 0.1f * matV1(0, 2);
        const float y2 = vc.y() - 0.1f * matV1(1, 2);
        const float z2 = vc.z() - 0.1f * matV1(2, 2);

        // Compute the numerator of the Equation (2)
        const float a012 = std::sqrt(
            ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
            * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
            + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
            * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
            + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
            * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

        // Compute the denominator of the Equation (2)
        const float l12 = std::sqrt(
            (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            + (z1 - z2) * (z1 - z2));

        // Compute the normal vector (la, lb, lc) of the edge line, i.e.,
        // the vector from the projection of the point (x0, y0, z0) on
        // the edge line between points (x1, y1, z1) and (x2, y2, z2)
        // to the point (x0, y0, z0)
        const float la = (
            (y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
            + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)))
            / a012 / l12;
        const float lb = -(
            (x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
            - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)))
            / a012 / l12;
        const float lc = -(
            (x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
            + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)))
            / a012 / l12;

        // Compute the point-to-line distance using the Equation (2),
        // i.e., the distance from the corner point in the current scan
        // (x0, y0, z0) to the edge lines between (x1, y1, z1) and
        // (x2, y2, z2) which are obtained from the corner points of
        // the map cloud
        const float ld2 = a012 / l12;

        const float s = 1.0f - 0.9f * std::fabs(ld2);

        if (s <= 0.1f)
            continue;

        // Compute the coefficient for the pose optimization
        pcl::PointXYZI coeff;
        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
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
            matA0(j, 0) = neighborPoint.x;
            matA0(j, 1) = neighborPoint.y;
            matA0(j, 2) = neighborPoint.z;
        }

        // Compute the normal vector (pa, pb, pc) that is perpenidcular to
        // the plane defined by a set of neighbor points `pointSearchInd`
        const Eigen::Vector3f matX0 = matA0.colPivHouseholderQr().solve(matB0);

        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1.0f;

        // Normalize the normal vector (pa, pb, pc) of the plane
        const float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // If all neighbor points `pointSearchInd` are on the same plane,
        // then the distance between the neighbor point and the plane
        // should be closer to zero
        bool planeValid = true;
        for (int j = 0; j < 5; ++j) {
            const pcl::PointXYZI& neighborPoint =
                this->_laserCloudSurfFromMap->at(pointSearchInd[j]);
            const float dist = pa * neighborPoint.x + pb * neighborPoint.y
                               + pc * neighborPoint.z + pd;

            if (std::abs(dist) > 0.2f) {
                planeValid = false;
                break;
            }
        }

        if (!planeValid)
            continue;

        // Compute the d_h using the Equation (3)
        // Note that the distance below could be negative
        const float pd2 = pa * pointSel.x + pb * pointSel.y
                          + pc * pointSel.z + pd;

        const float s = 1.0f - 0.9f * std::fabs(pd2)
                        / std::sqrt(calcPointDistance(pointSel));

        if (s <= 0.1f)
            continue;

        pcl::PointXYZI coeff;
        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        this->_laserCloudOri.push_back(pointOri);
        this->_coeffSel.push_back(coeff);
    }
}

} // namespace loam
