
// MultiScanRegistration.h

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

#ifndef LOAM_MULTI_SCAN_REGISTRATION_H
#define LOAM_MULTI_SCAN_REGISTRATION_H

#include "loam_velodyne/ScanRegistration.h"

#include <sensor_msgs/PointCloud2.h>

namespace loam {

/** \brief Class realizing a linear mapping from vertical point angle
 * to the corresponding scan ring.
 *
 */
class MultiScanMapper
{
public:
    /** \brief Construct a new multi scan mapper instance.
     *
     * @param lowerBound The lower vertical bound (degrees)
     * @param upperBound The upper vertical bound (degrees)
     * @param nScanRings The number of scan rings
     */
    MultiScanMapper(const float lowerBound = -15,
                    const float upperBound = 15,
                    const std::uint16_t nScanRings = 16);

    inline float getLowerBound() const
    { return this->_lowerBound; }
    inline float getUpperBound() const
    { return this->_upperBound; }
    inline std::uint16_t getNumberOfScanRings() const
    { return this->_nScanRings; }

    /** \brief Set mapping parameters.
     *
     * @param lowerBound The lower vertical bound (degrees)
     * @param upperBound The upper vertical bound (degrees)
     * @param nScanRings The number of scan rings
     */
    void set(const float lowerBound,
             const float upperBound,
             const std::uint16_t nScanRings);

    /** \brief Map the specified vertical point angle to its ring ID.
     *
     * @param angle The vertical point angle (in rad)
     * @return The ring ID
     */
    inline int getRingForAngle(const float angle) const
    { return static_cast<int>(((angle * 180.0f / M_PI) - this->_lowerBound)
                              * this->_factor + 0.5f); }

    /** Multi scan mapper for Velodyne VLP-16 according to data sheet. */
    static inline MultiScanMapper Velodyne_VLP_16()
    { return MultiScanMapper(-15.0, 15.0, 16); };
    /** Multi scan mapper for Velodyne HDL-32 according to data sheet. */
    static inline MultiScanMapper Velodyne_HDL_32()
    { return MultiScanMapper(-30.67f, 10.67f, 32); };
    /** Multi scan mapper for Velodyne HDL-64E according to data sheet. */
    static inline MultiScanMapper Velodyne_HDL_64E()
    { return MultiScanMapper(-24.9f, 2.0f, 64); };

private:
    // The vertical angle of the first scan ring
    float         _lowerBound;
    // The vertical angle of the last scan ring
    float         _upperBound;
    // Number of scan rings
    std::uint16_t _nScanRings;
    // Linear interpolation factor
    float         _factor;
};

/** \brief Class for registering point clouds received from multi-laser lidars.
 *
 */
class MultiScanRegistration : virtual public ScanRegistration
{
public:
    MultiScanRegistration(
        const MultiScanMapper& scanMapper = MultiScanMapper());

    bool setup(ros::NodeHandle& node, ros::NodeHandle& privateNode);

    /** \brief Handler method for input cloud messages.
     *
     * @param laserCloudMsg The new input cloud message to process
     */
    void handleCloudMessage(
        const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg);

    private:
    /** \brief Setup component in active mode.
     *
     * @param node The ROS node handle
     * @param privateNode The private ROS node handle
     */
    bool setupROS(ros::NodeHandle& node,
                  ros::NodeHandle& privateNode,
                  RegistrationParams& configOut) override;

    /** \brief Process a new input cloud.
     *
     * @param laserCloudIn The new input cloud to process
     * @param scanTime The scan (message) timestamp
     */
    void process(const pcl::PointCloud<pcl::PointXYZ>& laserCloudIn,
                 const Time& scanTime);

private:
    // System startup delay counter
    int _systemDelay = 20;
    // Mapper for mapping vertical point angles to scan ring IDs
    MultiScanMapper _scanMapper;
    // Full resolution point cloud grouped by rings (vertical angles)
    std::vector<pcl::PointCloud<pcl::PointXYZI>> _laserCloudScans;
    // Input cloud message subscriber
    ros::Subscriber _subLaserCloud;
};

} // namespace loam

#endif // LOAM_MULTI_SCAN_REGISTRATION_H
