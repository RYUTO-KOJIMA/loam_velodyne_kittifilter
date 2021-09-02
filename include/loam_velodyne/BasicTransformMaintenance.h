
// BasicTransformMaintenance.h

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

#include "Twist.h"

namespace loam {

/** \brief Rotate the given vector around y, x, and z axes
 * \param */

/** \brief Implementation of the LOAM transformation maintenance component. */
class BasicTransformMaintenance
{
public:
    void updateOdometry(double pitch, double yaw, double roll,
                        double x, double y, double z);

    void updateMappingTransform(const Twist& transformAftMapped,
                                const Twist& transformBefMapped);

    void updateMappingTransform(
        double pitch, double yaw, double roll,
        double x, double y, double z,
        double twistRotX, double twistRotY, double twistRotZ,
        double twistPosX, double twistPosY, double twistPosZ);

    // Combine the results from odometry and mapping
    void transformAssociateToMap();

    // Get the pose computed from odometry and mapping results
    const auto& transformMapped() const { return this->_transformMapped; }

private:
    // Current odometry pose in the global coordinate frame, i.e., pose of
    // the current odometry frame (/laser_odom) from the global coordinate
    // frame (camera_init)
    float _transformSum[6] { };
    // Update of the odometry pose (difference between the last and
    // current odometry poses) in the current odometry frame (/laser_odom),
    // i.e., pose of the last odometry frame from the current odometry
    // frame (/laser_odom)
    float _transformIncre[6] { };
    // Current pose in the global coordinate frame obtained by combining
    // results from odometry and mapping, i.e., pose of the current frame
    // (/camera) from the global coordinate frame (camera_init)
    float _transformMapped[6] { };
    // Last odometry pose in the global coordinate frame, i.e., pose of
    // the last odometry frame (/laser_odom) from the global coordinate
    // frame (camera_init)
    float _transformBefMapped[6] { };
    // Last mapped pose in the global coordinate frame, i.e., pose of
    // the last mapping frame (/aft_mapped) from the global coordinate
    // frame (camera_init)
    float _transformAftMapped[6] { };
};

} // namespace loam
