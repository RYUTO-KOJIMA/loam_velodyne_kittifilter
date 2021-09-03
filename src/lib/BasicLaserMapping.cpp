
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

bool BasicLaserMapping::process(const Time& laserOdometryTime)
{
   // skip some frames?!?
   _frameCount++;
   if (_frameCount < _stackFrameNum)
   {
      return false;
   }
   _frameCount = 0;
   // `_laserOdometryTime` is actually t_(k + 1) in the paper (Section VI),
   // i.e., the timestamp of the current scan (or current sweep, since each
   // sweep contains only one scan in this implementation)
   _laserOdometryTime = laserOdometryTime;

   pcl::PointXYZI pointSel;

   // relate incoming data to map
   // Compute the poses of the scans at t_(k + 1) and t_(k + 2) in the mapped
   // coordinate frame, i.e., `_transformAftMapped` and `_transformTobeMapped`
   transformAssociateToMap();

   // `_laserCloudCornerLast` and `_laserCloudSurfLast` are the sets of
   // corner and planar points in the scan at time t_(k + 1), and points in
   // `_laserCloudCornerLast` and `_laserCloudSurfLast` are reprojected to
   // the t_(k + 2), and `_transformTobeMapped` is used to transform the
   // coordinate at t_(k + 2) to the mapped coordinate frame

   for (auto const& pt : _laserCloudCornerLast->points)
   {
      pointAssociateToMap(pt, pointSel);
      _laserCloudCornerStack->push_back(pointSel);
   }

   for (auto const& pt : _laserCloudSurfLast->points)
   {
      pointAssociateToMap(pt, pointSel);
      _laserCloudSurfStack->push_back(pointSel);
   }

   pcl::PointXYZI pointOnYAxis;
   pointOnYAxis.x = 0.0;
   pointOnYAxis.y = 10.0;
   pointOnYAxis.z = 0.0;
   pointAssociateToMap(pointOnYAxis, pointOnYAxis);

   // `CUBE_SIZE` and `CUBE_HALF` are in centimeters (50cm and 25cm)
   auto const CUBE_SIZE = 50.0;
   auto const CUBE_HALF = CUBE_SIZE / 2;

   // Compute the index of the center cube in the 10m cubic area, and
   // each cube stores the point cloud
   // `_laserCloudCenWidth`, `_laserCloudCenHeight`, and `_laserCloudCenDepth`
   // are initially set to 10, 5, and 10, respectively, and the number of
   // the cubes, i.e., `_laserCloudWidth`, `_laserCloudHeight`, and
   // `_laserCloudDepth` are initially set to 21, 11, and 21
   int centerCubeI = int((_transformTobeMapped.pos.x() + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenWidth;
   int centerCubeJ = int((_transformTobeMapped.pos.y() + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenHeight;
   int centerCubeK = int((_transformTobeMapped.pos.z() + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenDepth;

   if (_transformTobeMapped.pos.x() + CUBE_HALF < 0) centerCubeI--;
   if (_transformTobeMapped.pos.y() + CUBE_HALF < 0) centerCubeJ--;
   if (_transformTobeMapped.pos.z() + CUBE_HALF < 0) centerCubeK--;

   // Slide the cubes in `_laserCloudCornerArray` and `_laserCloudSurfArray`
   // along X axis to constrain the `centerCubeI` to be within the range of
   // [3, `_laserCloudWidth` - 3)
   while (centerCubeI < 3)
   {
      for (int j = 0; j < _laserCloudHeight; j++)
      {
         for (int k = 0; k < _laserCloudDepth; k++)
         {
            for (int i = _laserCloudWidth - 1; i >= 1; i--)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i - 1, j, k);
               std::swap(_laserCloudCornerArray[indexA], _laserCloudCornerArray[indexB]);
               std::swap(_laserCloudSurfArray[indexA], _laserCloudSurfArray[indexB]);
            }
            const size_t indexC = toIndex(0, j, k);
            _laserCloudCornerArray[indexC]->clear();
            _laserCloudSurfArray[indexC]->clear();
         }
      }
      centerCubeI++;
      _laserCloudCenWidth++;
   }

   while (centerCubeI >= _laserCloudWidth - 3)
   {
      for (int j = 0; j < _laserCloudHeight; j++)
      {
         for (int k = 0; k < _laserCloudDepth; k++)
         {
            for (int i = 0; i < _laserCloudWidth - 1; i++)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i + 1, j, k);
               std::swap(_laserCloudCornerArray[indexA], _laserCloudCornerArray[indexB]);
               std::swap(_laserCloudSurfArray[indexA], _laserCloudSurfArray[indexB]);
            }
            const size_t indexC = toIndex(_laserCloudWidth - 1, j, k);
            _laserCloudCornerArray[indexC]->clear();
            _laserCloudSurfArray[indexC]->clear();
         }
      }
      centerCubeI--;
      _laserCloudCenWidth--;
   }

   // Slide the cubes in `_laserCloudCornerArray` and `_laserCloudSurfArray`
   // along Y axis to constrain the `centerCubeJ` to be within the range of
   // [3, `_laserCloudHeight` - 3)
   while (centerCubeJ < 3)
   {
      for (int i = 0; i < _laserCloudWidth; i++)
      {
         for (int k = 0; k < _laserCloudDepth; k++)
         {
            for (int j = _laserCloudHeight - 1; j >= 1; j--)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i, j - 1, k);
               std::swap(_laserCloudCornerArray[indexA], _laserCloudCornerArray[indexB]);
               std::swap(_laserCloudSurfArray[indexA], _laserCloudSurfArray[indexB]);
            }
            const size_t indexC = toIndex(i, 0, k);
            _laserCloudCornerArray[indexC]->clear();
            _laserCloudSurfArray[indexC]->clear();
         }
      }
      centerCubeJ++;
      _laserCloudCenHeight++;
   }

   while (centerCubeJ >= _laserCloudHeight - 3)
   {
      for (int i = 0; i < _laserCloudWidth; i++)
      {
         for (int k = 0; k < _laserCloudDepth; k++)
         {
            for (int j = 0; j < _laserCloudHeight - 1; j++)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i, j + 1, k);
               std::swap(_laserCloudCornerArray[indexA], _laserCloudCornerArray[indexB]);
               std::swap(_laserCloudSurfArray[indexA], _laserCloudSurfArray[indexB]);
            }
            const size_t indexC = toIndex(i, _laserCloudHeight - 1, k);
            _laserCloudCornerArray[indexC]->clear();
            _laserCloudSurfArray[indexC]->clear();
         }
      }
      centerCubeJ--;
      _laserCloudCenHeight--;
   }

   // Slide the cubes in `_laserCloudCornerArray` and `_laserCloudSurfArray`
   // along Z axis to constrain the `centerCubeK` to be within the range of
   // [3, `_laserCloudDepth` - 3)
   while (centerCubeK < 3)
   {
      for (int i = 0; i < _laserCloudWidth; i++)
      {
         for (int j = 0; j < _laserCloudHeight; j++)
         {
            for (int k = _laserCloudDepth - 1; k >= 1; k--)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i, j, k - 1);
               std::swap(_laserCloudCornerArray[indexA], _laserCloudCornerArray[indexB]);
               std::swap(_laserCloudSurfArray[indexA], _laserCloudSurfArray[indexB]);
            }
            const size_t indexC = toIndex(i, j, 0);
            _laserCloudCornerArray[indexC]->clear();
            _laserCloudSurfArray[indexC]->clear();
         }
      }
      centerCubeK++;
      _laserCloudCenDepth++;
   }

   while (centerCubeK >= _laserCloudDepth - 3)
   {
      for (int i = 0; i < _laserCloudWidth; i++)
      {
         for (int j = 0; j < _laserCloudHeight; j++)
         {
            for (int k = 0; k < _laserCloudDepth - 1; k++)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i, j, k + 1);
               std::swap(_laserCloudCornerArray[indexA], _laserCloudCornerArray[indexB]);
               std::swap(_laserCloudSurfArray[indexA], _laserCloudSurfArray[indexB]);
            }
            const size_t indexC = toIndex(i, j, _laserCloudDepth - 1);
            _laserCloudCornerArray[indexC]->clear();
            _laserCloudSurfArray[indexC]->clear();
         }
      }
      centerCubeK--;
      _laserCloudCenDepth--;
   }

   // `_laserCloudValidInd` and `_laserCloudSurroundInd` contain 125 cube
   // indices at most when all cubes around the center cube (i, j, k)
   // are in the field of view, or all cubes have valid indices
   _laserCloudValidInd.clear();
   _laserCloudSurroundInd.clear();
   for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
   {
      for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
      {
         for (int k = centerCubeK - 2; k <= centerCubeK + 2; k++)
         {
            if (i >= 0 && i < _laserCloudWidth &&
                j >= 0 && j < _laserCloudHeight &&
                k >= 0 && k < _laserCloudDepth)
            {
               // Convert the voxel index to the mapped coordinate frame
               float centerX = 50.0f * (i - _laserCloudCenWidth);
               float centerY = 50.0f * (j - _laserCloudCenHeight);
               float centerZ = 50.0f * (k - _laserCloudCenDepth);

               pcl::PointXYZI transform_pos = (pcl::PointXYZI) _transformTobeMapped.pos;

               // `corner` is the corner points of the cube at index (i, j, k)
               // in the mapped coordinate frame
               bool isInLaserFOV = false;
               for (int ii = -1; ii <= 1; ii += 2)
               {
                  for (int jj = -1; jj <= 1; jj += 2)
                  {
                     for (int kk = -1; kk <= 1; kk += 2)
                     {
                        pcl::PointXYZI corner;
                        corner.x = centerX + 25.0f * ii;
                        corner.y = centerY + 25.0f * jj;
                        corner.z = centerZ + 25.0f * kk;

                        float squaredSide1 = calcSquaredDiff(transform_pos, corner);
                        float squaredSide2 = calcSquaredDiff(pointOnYAxis, corner);

                        // `100.0f + squaredSide1 - squaredSide2` equals to
                        // `2 * 10 * sqrt(squaredSide1) * cos(x)` using law of
                        // cosines, where `x` is `90 - (vertical angle)`

                        float check1 = 100.0f + squaredSide1 - squaredSide2
                           - 10.0f * sqrt(3.0f) * sqrt(squaredSide1);

                        float check2 = 100.0f + squaredSide1 - squaredSide2
                           + 10.0f * sqrt(3.0f) * sqrt(squaredSide1);

                        // This holds if |100.0f + side1 - side2| is less than
                        // 10.0f * sqrt(3.0f) * sqrt(side1), which means that
                        // the vertical angle of the point is within the range
                        // of [-60, 60] (cos(x) is less than sqrt(3) / 2
                        // and is larger than -sqrt(3) / 2, i.e., x is larger
                        // than 30 degrees and is less than 150 degrees)
                        if (check1 < 0 && check2 > 0)
                        {
                           isInLaserFOV = true;
                        }
                     }
                  }
               }

               size_t cubeIdx = i + _laserCloudWidth * j + _laserCloudWidth * _laserCloudHeight * k;
               if (isInLaserFOV)
               {
                  _laserCloudValidInd.push_back(cubeIdx);
               }
               _laserCloudSurroundInd.push_back(cubeIdx);
            }
         }
      }
   }

   // prepare valid map corner and surface cloud for pose optimization
   _laserCloudCornerFromMap->clear();
   _laserCloudSurfFromMap->clear();
   for (auto const& ind : _laserCloudValidInd)
   {
      *_laserCloudCornerFromMap += *_laserCloudCornerArray[ind];
      *_laserCloudSurfFromMap += *_laserCloudSurfArray[ind];
   }

   // prepare feature stack clouds for pose optimization
   // Convert the point coordinates from the mapped coordinate frame to the
   // scan coordinate frame at t_(k + 2)
   for (auto& pt : *_laserCloudCornerStack)
      pointAssociateTobeMapped(pt, pt);

   for (auto& pt : *_laserCloudSurfStack)
      pointAssociateTobeMapped(pt, pt);

   // down sample feature stack clouds
   _laserCloudCornerStackDS->clear();
   _downSizeFilterCorner.setInputCloud(_laserCloudCornerStack);
   _downSizeFilterCorner.filter(*_laserCloudCornerStackDS);
   size_t laserCloudCornerStackNum = _laserCloudCornerStackDS->size();

   _laserCloudSurfStackDS->clear();
   _downSizeFilterSurf.setInputCloud(_laserCloudSurfStack);
   _downSizeFilterSurf.filter(*_laserCloudSurfStackDS);
   size_t laserCloudSurfStackNum = _laserCloudSurfStackDS->size();

   _laserCloudCornerStack->clear();
   _laserCloudSurfStack->clear();

   // run pose optimization
   optimizeTransformTobeMapped();

   // store down sized corner stack points in corresponding cube clouds
   for (int i = 0; i < laserCloudCornerStackNum; i++)
   {
      // Convert the point coordinates from the scan frame to the map frame
      pointAssociateToMap(_laserCloudCornerStackDS->points[i], pointSel);

      // Compute the index of the cube corresponding to the point
      int cubeI = int((pointSel.x + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenWidth;
      int cubeJ = int((pointSel.y + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenHeight;
      int cubeK = int((pointSel.z + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenDepth;

      if (pointSel.x + CUBE_HALF < 0) cubeI--;
      if (pointSel.y + CUBE_HALF < 0) cubeJ--;
      if (pointSel.z + CUBE_HALF < 0) cubeK--;

      // Append the aligned point to the cube
      if (cubeI >= 0 && cubeI < _laserCloudWidth &&
          cubeJ >= 0 && cubeJ < _laserCloudHeight &&
          cubeK >= 0 && cubeK < _laserCloudDepth)
      {
         size_t cubeInd = cubeI + _laserCloudWidth * cubeJ + _laserCloudWidth * _laserCloudHeight * cubeK;
         _laserCloudCornerArray[cubeInd]->push_back(pointSel);
      }
   }

   // store down sized surface stack points in corresponding cube clouds
   for (int i = 0; i < laserCloudSurfStackNum; i++)
   {
      pointAssociateToMap(_laserCloudSurfStackDS->points[i], pointSel);

      int cubeI = int((pointSel.x + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenWidth;
      int cubeJ = int((pointSel.y + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenHeight;
      int cubeK = int((pointSel.z + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenDepth;

      if (pointSel.x + CUBE_HALF < 0) cubeI--;
      if (pointSel.y + CUBE_HALF < 0) cubeJ--;
      if (pointSel.z + CUBE_HALF < 0) cubeK--;

      if (cubeI >= 0 && cubeI < _laserCloudWidth &&
          cubeJ >= 0 && cubeJ < _laserCloudHeight &&
          cubeK >= 0 && cubeK < _laserCloudDepth)
      {
         size_t cubeInd = cubeI + _laserCloudWidth * cubeJ + _laserCloudWidth * _laserCloudHeight * cubeK;
         _laserCloudSurfArray[cubeInd]->push_back(pointSel);
      }
   }

   // down size all valid (within field of view) feature cube clouds
   for (auto const& ind : _laserCloudValidInd)
   {
      _laserCloudCornerDSArray[ind]->clear();
      _downSizeFilterCorner.setInputCloud(_laserCloudCornerArray[ind]);
      _downSizeFilterCorner.filter(*_laserCloudCornerDSArray[ind]);

      _laserCloudSurfDSArray[ind]->clear();
      _downSizeFilterSurf.setInputCloud(_laserCloudSurfArray[ind]);
      _downSizeFilterSurf.filter(*_laserCloudSurfDSArray[ind]);

      // swap cube clouds for next processing
      _laserCloudCornerArray[ind].swap(_laserCloudCornerDSArray[ind]);
      _laserCloudSurfArray[ind].swap(_laserCloudSurfDSArray[ind]);
   }

   transformFullResToMap();
   _downsizedMapCreated = createDownsizedMap();

   return true;
}


void BasicLaserMapping::updateIMU(IMUState2 const& newState)
{
   _imuHistory.push(newState);
}

void BasicLaserMapping::updateOdometry(double pitch, double yaw, double roll, double x, double y, double z)
{
   _transformSum.rot_x = pitch;
   _transformSum.rot_y = yaw;
   _transformSum.rot_z = roll;

   _transformSum.pos.x() = float(x);
   _transformSum.pos.y() = float(y);
   _transformSum.pos.z() = float(z);
}

void BasicLaserMapping::updateOdometry(Twist const& twist)
{
   _transformSum = twist;
}

nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerFromMap;
nanoflann::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfFromMap;

void BasicLaserMapping::optimizeTransformTobeMapped()
{
   if (_laserCloudCornerFromMap->size() <= 10 || _laserCloudSurfFromMap->size() <= 100)
      return;

   pcl::PointXYZI pointSel, pointOri, /*pointProj, */coeff;

   std::vector<int> pointSearchInd(5, 0);
   std::vector<float> pointSearchSqDis(5, 0);

   kdtreeCornerFromMap.setInputCloud(_laserCloudCornerFromMap);
   kdtreeSurfFromMap.setInputCloud(_laserCloudSurfFromMap);

   Eigen::Matrix<float, 5, 3> matA0;
   Eigen::Matrix<float, 5, 1> matB0;
   Eigen::Vector3f matX0;
   Eigen::Matrix3f matA1;
   Eigen::Matrix<float, 1, 3> matD1;
   Eigen::Matrix3f matV1;

   matA0.setZero();
   matB0.setConstant(-1);
   matX0.setZero();

   matA1.setZero();
   matD1.setZero();
   matV1.setZero();

   bool isDegenerate = false;
   Eigen::Matrix<float, 6, 6> matP;

   size_t laserCloudCornerStackNum = _laserCloudCornerStackDS->size();
   size_t laserCloudSurfStackNum = _laserCloudSurfStackDS->size();

   // Start the iterations of the Gauss-Newton method
   for (size_t iterCount = 0; iterCount < _maxIterations; iterCount++)
   {
      _laserCloudOri.clear();
      _coeffSel.clear();

      // For each corner point in the downsampled current point cloud,
      // find the closest neighbors in the map cloud
      for (int i = 0; i < laserCloudCornerStackNum; i++)
      {
         pointOri = _laserCloudCornerStackDS->points[i];
         // Convert the corner point coordinates in the scan coordinate frame
         // at t_(k + 2) to the mapped coordinate frame using the current pose
         // estimate, i.e., `_transformTobeMapped`
         pointAssociateToMap(pointOri, pointSel);
         // Find the 5 closest neighbors in the map cloud
         kdtreeCornerFromMap.nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

         // If distances to all closest neighbors are less than 1m, then
         // compute the coefficient for the Gauss-Newton optimization
         if (pointSearchSqDis[4] < 1.0)
         {
            // Compute the average of the closest neighbor coordinates
            Vector3 vc(0, 0, 0);

            for (int j = 0; j < 5; j++)
               vc += Vector3(_laserCloudCornerFromMap->points[pointSearchInd[j]]);
            vc /= 5.0;

            // Compute the lower-triangular part of the covariance matrix of
            // the closest neighbor coordinates and then compute eigenvectors
            // and eigenvalues of the covariance matrix
            Eigen::Matrix3f mat_a;
            mat_a.setZero();

            for (int j = 0; j < 5; j++)
            {
               Vector3 a = Vector3(_laserCloudCornerFromMap->points[pointSearchInd[j]]) - vc;

               mat_a(0, 0) += a.x() * a.x();
               mat_a(1, 0) += a.x() * a.y();
               mat_a(2, 0) += a.x() * a.z();
               mat_a(1, 1) += a.y() * a.y();
               mat_a(2, 1) += a.y() * a.z();
               mat_a(2, 2) += a.z() * a.z();
            }
            matA1 = mat_a / 5.0;
            // This solver only looks at the lower-triangular part of matA1.
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> esolver(matA1);
            // Eigenvalues are sorted in an ascending order
            matD1 = esolver.eigenvalues().real();
            matV1 = esolver.eigenvectors().real();

            // If one eigenvalue is larger than the other two, then the closest
            // neighbors represent the edge line, and the eigenvector associated
            // with the largest eigenvalue represents the orientation of the
            // edge line
            if (matD1(0, 2) > 3 * matD1(0, 1))
            {
               float x0 = pointSel.x;
               float y0 = pointSel.y;
               float z0 = pointSel.z;
               // The position of the edge line is the center of the 5 closest
               // neighbors that represent the edge line, and two points
               // (x1, y1, z1) and (x2, y2, z2) should be on the edge line
               float x1 = vc.x() + 0.1 * matV1(0, 2);
               float y1 = vc.y() + 0.1 * matV1(1, 2);
               float z1 = vc.z() + 0.1 * matV1(2, 2);
               float x2 = vc.x() - 0.1 * matV1(0, 2);
               float y2 = vc.y() - 0.1 * matV1(1, 2);
               float z2 = vc.z() - 0.1 * matV1(2, 2);

               // Compute the numerator of the Equation (2)
               float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                 * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                 + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                 * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                 + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                 * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

               // Compute the denominator of the Equation (2)
               float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

               // Compute the normal vector (la, lb, lc) of the edge line, i.e.,
               // the vector from the projection of the point (x0, y0, z0) on
               // the edge line between points (x1, y1, z1) and (x2, y2, z2)
               // to the point (x0, y0, z0)
               float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                           + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

               float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                            - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

               float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                            + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

               // Compute the point-to-line distance using the Equation (2),
               // i.e., the distance from the corner point in the current scan
               // (x0, y0, z0) to the edge lines between (x1, y1, z1) and
               // (x2, y2, z2) which are obtained from the corner points of
               // the map cloud
               float ld2 = a012 / l12;

//                // TODO: Why writing to a variable that's never read? Maybe it should be used afterwards?
//                pointProj = pointSel;
//                pointProj.x -= la * ld2;
//                pointProj.y -= lb * ld2;
//                pointProj.z -= lc * ld2;

               float s = 1 - 0.9f * fabs(ld2);

               // Compute the coefficient for the pose optimization
               coeff.x = s * la;
               coeff.y = s * lb;
               coeff.z = s * lc;
               coeff.intensity = s * ld2;

               if (s > 0.1)
               {
                  _laserCloudOri.push_back(pointOri);
                  _coeffSel.push_back(coeff);
               }
            }
         }
      }

      // For each planar point in the downsampled current point cloud,
      // find the closest neighbors in the map cloud
      for (int i = 0; i < laserCloudSurfStackNum; i++)
      {
         pointOri = _laserCloudSurfStackDS->points[i];
         // Convert the planar point coordinates in the scan coordinate frame
         // at t_(k + 2) to the mapped coordinate frame using the current pose
         // estimate
         pointAssociateToMap(pointOri, pointSel);
         // Find the 5 closest neighbors in the map cloud
         kdtreeSurfFromMap.nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

         // If distances to all closest neighbors are less than 1m, then
         // compute the coefficient for the Gauss-Newton optimization
         if (pointSearchSqDis[4] < 1.0)
         {
            // Store coordinates of the closest neighbors to the matrix
            for (int j = 0; j < 5; j++)
            {
               matA0(j, 0) = _laserCloudSurfFromMap->points[pointSearchInd[j]].x;
               matA0(j, 1) = _laserCloudSurfFromMap->points[pointSearchInd[j]].y;
               matA0(j, 2) = _laserCloudSurfFromMap->points[pointSearchInd[j]].z;
            }

            // Compute the normal vector (pa, pb, pc) that is perpenidcular to
            // the plane defined by a set of neighbor points `pointSearchInd`
            matX0 = matA0.colPivHouseholderQr().solve(matB0);

            float pa = matX0(0, 0);
            float pb = matX0(1, 0);
            float pc = matX0(2, 0);
            float pd = 1;

            // Normalize the normal vector (pa, pb, pc) of the plane
            float ps = sqrt(pa * pa + pb * pb + pc * pc);
            pa /= ps;
            pb /= ps;
            pc /= ps;
            pd /= ps;

            // If all neighbor points `pointSearchInd` are on the same plane,
            // then the below fabs() should be closer to zero for all points
            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
               if (fabs(pa * _laserCloudSurfFromMap->points[pointSearchInd[j]].x +
                        pb * _laserCloudSurfFromMap->points[pointSearchInd[j]].y +
                        pc * _laserCloudSurfFromMap->points[pointSearchInd[j]].z + pd) > 0.2)
               {
                  planeValid = false;
                  break;
               }
            }

            if (planeValid)
            {
               // Compute the d_h using the Equation (3)
               // Note that the distance below could be negative
               float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

               //                // TODO: Why writing to a variable that's never read? Maybe it should be used afterwards?
               //                pointProj = pointSel;
               //                pointProj.x -= pa * pd2;
               //                pointProj.y -= pb * pd2;
               //                pointProj.z -= pc * pd2;

               float s = 1 - 0.9f * fabs(pd2) / sqrt(calcPointDistance(pointSel));

               coeff.x = s * pa;
               coeff.y = s * pb;
               coeff.z = s * pc;
               coeff.intensity = s * pd2;

               if (s > 0.1)
               {
                  _laserCloudOri.push_back(pointOri);
                  _coeffSel.push_back(coeff);
               }
            }
         }
      }

      float srx = _transformTobeMapped.rot_x.sin();
      float crx = _transformTobeMapped.rot_x.cos();
      float sry = _transformTobeMapped.rot_y.sin();
      float cry = _transformTobeMapped.rot_y.cos();
      float srz = _transformTobeMapped.rot_z.sin();
      float crz = _transformTobeMapped.rot_z.cos();

      size_t laserCloudSelNum = _laserCloudOri.size();
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

      for (int i = 0; i < laserCloudSelNum; i++)
      {
         pointOri = _laserCloudOri.points[i];
         coeff = _coeffSel.points[i];

         float arx = (crx*sry*srz*pointOri.x + crx * crz*sry*pointOri.y - srx * sry*pointOri.z) * coeff.x
            + (-srx * srz*pointOri.x - crz * srx*pointOri.y - crx * pointOri.z) * coeff.y
            + (crx*cry*srz*pointOri.x + crx * cry*crz*pointOri.y - cry * srx*pointOri.z) * coeff.z;

         float ary = ((cry*srx*srz - crz * sry)*pointOri.x
                      + (sry*srz + cry * crz*srx)*pointOri.y + crx * cry*pointOri.z) * coeff.x
            + ((-cry * crz - srx * sry*srz)*pointOri.x
               + (cry*srz - crz * srx*sry)*pointOri.y - crx * sry*pointOri.z) * coeff.z;

         float arz = ((crz*srx*sry - cry * srz)*pointOri.x + (-cry * crz - srx * sry*srz)*pointOri.y)*coeff.x
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

      if (iterCount == 0)
      {
         // Check the occurrence of the degeneration following the paper:
         // Ji Zhang, Michael Kaess, and Sanjiv Singh. "On Degeneracy of
         // Optimization-based State Estimation Problems," in the Proceedings
         // of the IEEE International Conference on Robotics and Automation
         // (ICRA), 2016.
         Eigen::Matrix<float, 1, 6> matE;
         Eigen::Matrix<float, 6, 6> matV;
         Eigen::Matrix<float, 6, 6> matV2;

         // Compute the eigenvalues and eigenvectors of the Hessian matrix
         Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float, 6, 6> > esolver(matAtA);
         matE = esolver.eigenvalues().real();
         matV = esolver.eigenvectors().real();

         matV2 = matV;

         isDegenerate = false;
         float eignThre[6] = { 100, 100, 100, 100, 100, 100 };
         // Eigenvalues are sorted in the increasing order
         // Detect the occurrence of the degeneration if one of the
         // eigenvalues is less than 100
         for (int i = 0; i < 6; i++)
         {
            if (matE(0, i) < eignThre[i])
            {
               for (int j = 0; j < 6; j++)
               {
                  matV2(i, j) = 0;
               }
               isDegenerate = true;
            }
            else
            {
               break;
            }
         }

        // Do not update the transformation along the degenerate direction
         matP = matV.inverse() * matV2;
      }

      if (isDegenerate)
      {
         // Do not update the transformation along the degenerate direction
         Eigen::Matrix<float, 6, 1> matX2(matX);
         matX = matP * matX2;
      }

      // Update the transformation (rotation and translation)
      _transformTobeMapped.rot_x += matX(0, 0);
      _transformTobeMapped.rot_y += matX(1, 0);
      _transformTobeMapped.rot_z += matX(2, 0);
      _transformTobeMapped.pos.x() += matX(3, 0);
      _transformTobeMapped.pos.y() += matX(4, 0);
      _transformTobeMapped.pos.z() += matX(5, 0);

      // Compute the increment in degrees and centimeters
      float deltaR = sqrt(pow(rad2deg(matX(0, 0)), 2) +
                          pow(rad2deg(matX(1, 0)), 2) +
                          pow(rad2deg(matX(2, 0)), 2));
      float deltaT = sqrt(pow(matX(3, 0) * 100, 2) +
                          pow(matX(4, 0) * 100, 2) +
                          pow(matX(5, 0) * 100, 2));

      // Terminate the Gauss-Newton method if the increment is small
      if (deltaR < _deltaRAbort && deltaT < _deltaTAbort)
         break;
   }

   // Refine the transformation using IMU data and update the transformation
   transformUpdate();
}

} // namespace loam
