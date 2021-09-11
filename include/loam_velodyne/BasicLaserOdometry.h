
// BasicLaserOdometry.h

#pragma once

#include "loam_velodyne/Twist.h"
#include "loam_velodyne/nanoflann_pcl.h"

#include "loam_velodyne/LaserOdometryMetrics.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace loam {

/** \brief Implementation of the LOAM laser odometry component. */
class BasicLaserOdometry
{
public:
    BasicLaserOdometry(float scanPeriod = 0.1, std::size_t maxIterations = 25);

    /** \brief Try to process buffered data. */
    void process();

    void updateIMU(const pcl::PointCloud<pcl::PointXYZ>& imuTrans);

    auto& cornerPointsSharp() { return this->_cornerPointsSharp; }
    auto& cornerPointsLessSharp() { return this->_cornerPointsLessSharp; }
    auto& surfPointsFlat() { return this->_surfPointsFlat; }
    auto& surfPointsLessFlat() { return this->_surfPointsLessFlat; }
    auto& laserCloud() { return this->_laserCloud; }

    const auto& transformSum() const { return this->_transformSum; }
    const auto& transform() const { return this->_transform; }
    const auto& lastCornerCloud() const { return this->_lastCornerCloud; }
    const auto& lastSurfaceCloud() const { return this->_lastSurfaceCloud; }

    void setScanPeriod(float val) { this->_scanPeriod = val; }
    void setMaxIterations(std::size_t val) { this->_maxIterations = val; }
    void setDeltaTAbort(float val) { this->_deltaTAbort = val; }
    void setDeltaRAbort(float val) { this->_deltaRAbort = val; }

    auto frameCount() const { return this->_frameCount; }
    auto scanPeriod() const { return this->_scanPeriod; }
    auto maxIterations() const { return this->_maxIterations; }
    auto deltaTAbort() const { return this->_deltaTAbort; }
    auto deltaRAbort() const { return this->_deltaRAbort; }

    /** \brief Transform the given point cloud to the end of the sweep.
     *
     * @param cloud The point cloud to transform
     */
    std::size_t transformToEnd(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

private:
    /** \brief Transform the given point to the start of the sweep.
     *
     * @param pi The point to transform
     * @param po The point instance for storing the result
     */
    pcl::PointXYZI transformToStart(const pcl::PointXYZI& pi);

    // Compute three rotation matrices, R_al = Ry(aly) Rx(alx) Rz(alz),
    // R_bl = Ry(bly) Rx(blx) Rz(blz), and R_bc = Ry(bcy) Rx(bcx) Rz(bcz)
    // and store three Euler angles (acx, acy, acz) that correspond to the
    // rotation matrix R = R_bc (R_bl)^T R_al
    void pluginIMURotation(const Angle& bcx, const Angle& bcy, const Angle& bcz,
                           const Angle& blx, const Angle& bly, const Angle& blz,
                           const Angle& alx, const Angle& aly, const Angle& alz,
                           Angle& acx, Angle& acy, Angle& acz);

    // Compute two rotation matrices, R_c = Ry(cy) Rx(cx) Rz(cz) and
    // R_l = Ry(ly) Rx(lx) Rz(lz) and store three Euler angles (ox, oy, oz)
    // that correspond to the rotation matrix R = R_c R_l
    void accumulateRotation(Angle cx, Angle cy, Angle cz,
                            Angle lx, Angle ly, Angle lz,
                            Angle& ox, Angle& oy, Angle& oz);

    // Perform the Gauss-Newton optimization and update the pose transformation
    void performOptimization();

    // Check the occurrence of the degeneration
    bool checkDegeneration(
        const Eigen::Matrix<float, 6, 6>& hessianMat,
        Eigen::Matrix<float, 6, 6>& projectionMat) const;

    // Compute the distances and coefficients from the point-to-edge
    // correspondences
    void computeCornerDistances(int iterCount);
    // Find point-to-edge correspondences from the corner point cloud
    void findCornerCorrespondence();
    // Compute the distances and coefficients from the point-to-plane
    // correspondences
    void computePlaneDistances(int iterCount);
    // Find point-to-plane correspondences from the planar point cloud
    void findPlaneCorrespondence();

protected:
    // Clear the metrics message
    void clearMetricsMsg();

protected:
    // Flag enabled if the input point clouds are undistorted
    // Point clouds in Kitti odometry dataset are undistorted
    bool _pointUndistorted;
    // Flag to publish the full-resolution point clouds
    bool _fullPointCloudPublished;

    // Flag to enable the metrics
    bool _metricsEnabled;
    // Metrics message
    loam_velodyne::LaserOdometryMetrics _metricsMsg;

private:
    // Time per scan
    float _scanPeriod;
    // Number of processed frames
    long _frameCount;
    // Maximum number of iterations
    std::size_t _maxIterations;
    // Initialization flag
    bool _systemInited;

    // Optimization abort threshold for deltaT
    float _deltaTAbort;
    // Optimization abort threshold for deltaR
    float _deltaRAbort;

    // Last corner points cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _lastCornerCloud;
    // Last surface points cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _lastSurfaceCloud;

    // Point selection
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloudOri;
    // Point selection coefficients
    pcl::PointCloud<pcl::PointXYZI>::Ptr _coeffSel;

    // Last corner cloud KD-tree
    nanoflann::KdTreeFLANN<pcl::PointXYZI> _lastCornerKDTree;
    // Last surface cloud KD-tree
    nanoflann::KdTreeFLANN<pcl::PointXYZI> _lastSurfaceKDTree;

    // Sharp corner points cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _cornerPointsSharp;
    // Less sharp corner points cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _cornerPointsLessSharp;
    // Flat surface points cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _surfPointsFlat;
    // Less flat surface points cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _surfPointsLessFlat;
    // Full resolution cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr _laserCloud;

    // First corner point search index buffer
    std::vector<int> _pointSearchCornerInd1;
    // Second corner point search index buffer
    std::vector<int> _pointSearchCornerInd2;

    // First surface point search index buffer
    std::vector<int> _pointSearchSurfInd1;
    // Second surface point search index buffer
    std::vector<int> _pointSearchSurfInd2;
    // Third surface point search index buffer
    std::vector<int> _pointSearchSurfInd3;

    // Optimized pose transformation
    Twist _transform;
    // Accumulated optimized pose transformation
    Twist _transformSum;

    Angle _imuRollStart;
    Angle _imuPitchStart;
    Angle _imuYawStart;
    Angle _imuRollEnd;
    Angle _imuPitchEnd;
    Angle _imuYawEnd;

    Vector3 _imuShiftFromStart;
    Vector3 _imuVeloFromStart;
};

} // namespace loam
