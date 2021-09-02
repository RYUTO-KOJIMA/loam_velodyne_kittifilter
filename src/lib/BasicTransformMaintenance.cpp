
// BasicTransformMaintenance.cpp

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

#include "loam_velodyne/BasicTransformMaintenance.h"
#include "loam_velodyne/Transform.hpp"

namespace loam {

void BasicTransformMaintenance::updateOdometry(
   double pitch, double yaw, double roll,
   double x, double y, double z)
{
    this->_transformSum[0] = static_cast<float>(pitch);
    this->_transformSum[1] = static_cast<float>(yaw);
    this->_transformSum[2] = static_cast<float>(roll);
    this->_transformSum[3] = static_cast<float>(x);
    this->_transformSum[4] = static_cast<float>(y);
    this->_transformSum[5] = static_cast<float>(z);
}

void BasicTransformMaintenance::updateMappingTransform(
    double pitch, double yaw, double roll,
    double x, double y, double z,
    double twistRotX, double twistRotY, double twistRotZ,
    double twistPosX, double twistPosY, double twistPosZ)
{
    this->_transformAftMapped[0] = static_cast<float>(pitch);
    this->_transformAftMapped[1] = static_cast<float>(yaw);
    this->_transformAftMapped[2] = static_cast<float>(roll);
    this->_transformAftMapped[3] = static_cast<float>(x);
    this->_transformAftMapped[4] = static_cast<float>(y);
    this->_transformAftMapped[5] = static_cast<float>(z);

    this->_transformBefMapped[0] = static_cast<float>(twistRotX);
    this->_transformBefMapped[1] = static_cast<float>(twistRotY);
    this->_transformBefMapped[2] = static_cast<float>(twistRotZ);
    this->_transformBefMapped[3] = static_cast<float>(twistPosX);
    this->_transformBefMapped[4] = static_cast<float>(twistPosY);
    this->_transformBefMapped[5] = static_cast<float>(twistPosZ);
}

void BasicTransformMaintenance::updateMappingTransform(
    const Twist& transformAftMapped, const Twist& transformBefMapped)
{
    this->updateMappingTransform(
        transformAftMapped.rot_x.rad(), transformAftMapped.rot_y.rad(),
        transformAftMapped.rot_z.rad(),
        transformAftMapped.pos.x(), transformAftMapped.pos.y(),
        transformAftMapped.pos.z(),
        transformBefMapped.rot_x.rad(), transformBefMapped.rot_y.rad(),
        transformBefMapped.rot_z.rad(),
        transformBefMapped.pos.x(), transformBefMapped.pos.y(),
        transformBefMapped.pos.z());
}

// Combine the results from odometry and mapping
void BasicTransformMaintenance::transformAssociateToMap()
{
    // This method is basically the same as `transformAssociateToMap()` in
    // `BasicLaserMapping`

    // Create rotation matrices from Euler angles in `_transformSum`,
    // `_transformBefMapped`, and `_transformAftMapped`
    const Eigen::Matrix3f rotationMatSum = rotationMatrixZXY(
        this->_transformSum[0], this->_transformSum[1],
        this->_transformSum[2]);
    const Eigen::Matrix3f rotationMatBefMapped = rotationMatrixZXY(
        this->_transformBefMapped[0], this->_transformBefMapped[1],
        this->_transformBefMapped[2]);
    const Eigen::Matrix3f rotationMatAftMapped = rotationMatrixZXY(
        this->_transformAftMapped[0], this->_transformAftMapped[1],
        this->_transformAftMapped[2]);

    // Compute the odometry pose update in a global coordinate frame
    const Eigen::Vector3f globalIncre {
        this->_transformBefMapped[3] - this->_transformSum[3],
        this->_transformBefMapped[4] - this->_transformSum[4],
        this->_transformBefMapped[5] - this->_transformSum[5] };
    // Compute the odometry pose update in a local odometry frame
    const Eigen::Vector3f transformIncre =
        rotationMatSum.transpose() * globalIncre;

    this->_transformIncre[3] = transformIncre.x();
    this->_transformIncre[4] = transformIncre.y();
    this->_transformIncre[5] = transformIncre.z();

    // Compose three rotation matrices above for `_transformMapped`
    const Eigen::Matrix3f rotationMatMapped =
        rotationMatAftMapped * rotationMatBefMapped.transpose()
        * rotationMatSum;
    // Get three Euler angles from the rotation matrix above
    Eigen::Vector3f eulerAnglesMapped;
    eulerAnglesFromRotationZXY(
        rotationMatMapped, eulerAnglesMapped.x(),
        eulerAnglesMapped.y(), eulerAnglesMapped.z());

    this->_transformMapped[0] = eulerAnglesMapped.x();
    this->_transformMapped[1] = eulerAnglesMapped.y();
    this->_transformMapped[2] = eulerAnglesMapped.z();

    // Combine the pose from odometry and mapping
    const Eigen::Vector3f transformAftMapped {
        this->_transformAftMapped[3], this->_transformAftMapped[4],
        this->_transformAftMapped[5] };
    const Eigen::Vector3f transformMapped =
        transformAftMapped - rotationMatMapped * transformIncre;

    this->_transformMapped[3] = transformMapped.x();
    this->_transformMapped[4] = transformMapped.y();
    this->_transformMapped[5] = transformMapped.z();

    return;
}

} // namespace loam
