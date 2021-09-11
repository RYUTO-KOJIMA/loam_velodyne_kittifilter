
// BasicLaserOdometry.cpp

#include "loam_velodyne/BasicLaserOdometry.h"
#include "loam_velodyne/Common.h"
#include "loam_velodyne/Transform.hpp"
#include "loam_velodyne/MathUtils.h"

#include <pcl/filters/filter.h>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace loam {

BasicLaserOdometry::BasicLaserOdometry(
    float scanPeriod, std::size_t maxIterations) :
    _residualScale(0.05f),
    _eigenThresholdTrans(10.0f),
    _eigenThresholdRot(10.0f),
    _weightDecayCorner(1.8f),
    _weightThresholdCorner(0.1f),
    _sqDistThresholdCorner(25.0f),
    _weightDecaySurface(1.8f),
    _weightThresholdSurface(0.1f),
    _sqDistThresholdSurface(25.0f),
    _pointUndistorted(false),
    _fullPointCloudPublished(false),
    _metricsEnabled(false),
    _scanPeriod(scanPeriod),
    _frameCount(0),
    _maxIterations(maxIterations),
    _systemInited(false),
    _deltaTAbort(0.1),
    _deltaRAbort(0.1),
    _cornerPointsSharp(new pcl::PointCloud<pcl::PointXYZI>()),
    _cornerPointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>()),
    _surfPointsFlat(new pcl::PointCloud<pcl::PointXYZI>()),
    _surfPointsLessFlat(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloud(new pcl::PointCloud<pcl::PointXYZI>()),
    _lastCornerCloud(new pcl::PointCloud<pcl::PointXYZI>()),
    _lastSurfaceCloud(new pcl::PointCloud<pcl::PointXYZI>()),
    _laserCloudOri(new pcl::PointCloud<pcl::PointXYZI>()),
    _coeffSel(new pcl::PointCloud<pcl::PointXYZI>())
{
}

pcl::PointXYZI BasicLaserOdometry::transformToStart(const pcl::PointXYZI& pi)
{
    const float relTime = pi.intensity - static_cast<int>(pi.intensity);
    const float s = this->_pointUndistorted ? 0.0f :
                    (1.0f / this->_scanPeriod) * relTime;

    pcl::PointXYZI po;
    po.x = pi.x - s * this->_transform.pos.x();
    po.y = pi.y - s * this->_transform.pos.y();
    po.z = pi.z - s * this->_transform.pos.z();
    po.intensity = pi.intensity;

    Angle rx = -s * this->_transform.rot_x.rad();
    Angle ry = -s * this->_transform.rot_y.rad();
    Angle rz = -s * this->_transform.rot_z.rad();
    rotateZXY(po, rz, rx, ry);

    return po;
}

std::size_t BasicLaserOdometry::transformToEnd(
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
    const std::size_t cloudSize = cloud->points.size();

    for (std::size_t i = 0; i < cloudSize; ++i) {
        pcl::PointXYZI& point = cloud->points[i];

        // Transform to the start of the sweep
        float relTime = point.intensity - static_cast<int>(point.intensity);
        float s = this->_pointUndistorted ? 0.0f :
                  (1.0f / this->_scanPeriod) * relTime;

        point.x -= s * this->_transform.pos.x();
        point.y -= s * this->_transform.pos.y();
        point.z -= s * this->_transform.pos.z();
        point.intensity = static_cast<int>(point.intensity);

        Angle rx = -s * this->_transform.rot_x.rad();
        Angle ry = -s * this->_transform.rot_y.rad();
        Angle rz = -s * this->_transform.rot_z.rad();
        rotateZXY(point, rz, rx, ry);

        // Then transform to the end of the sweep
        rotateYXZ(point,
                  this->_transform.rot_y,
                  this->_transform.rot_x,
                  this->_transform.rot_z);

        // Then transform to consider the nonlinear motion, i.e., motion
        // caused by acceleration or deceleration
        point.x += this->_transform.pos.x() - this->_imuShiftFromStart.x();
        point.y += this->_transform.pos.y() - this->_imuShiftFromStart.y();
        point.z += this->_transform.pos.z() - this->_imuShiftFromStart.z();

        rotateZXY(point,
                  this->_imuRollStart,
                  this->_imuPitchStart,
                  this->_imuYawStart);
        rotateYXZ(point,
                  -this->_imuYawEnd,
                  -this->_imuPitchEnd,
                  -this->_imuRollEnd);
    }

    return cloudSize;
}

void BasicLaserOdometry::pluginIMURotation(
    const Angle& bcx, const Angle& bcy, const Angle& bcz,
    const Angle& blx, const Angle& bly, const Angle& blz,
    const Angle& alx, const Angle& aly, const Angle& alz,
    Angle& acx, Angle& acy, Angle& acz)
{
    // Create rotation matrices R_bc = Ry(bcy) Rx(bcx) Rz(bcz),
    // R_bl = Ry(bly) Rx(blx) Rz(blz), and R_al = Ry(aly) Rx(alx) Rz(alz)
    const Eigen::Matrix3f rotationMatBc =
        rotationMatrixZXY(bcx.rad(), bcy.rad(), bcz.rad());
    const Eigen::Matrix3f rotationMatBl =
        rotationMatrixZXY(blx.rad(), bly.rad(), blz.rad());
    const Eigen::Matrix3f rotationMatAl =
        rotationMatrixZXY(alx.rad(), aly.rad(), alz.rad());

    // Compose three rotation matrices
    const Eigen::Matrix3f rotationMatAc =
        rotationMatBc * rotationMatBl.transpose() * rotationMatAl;
    // Get three Euler angles from the rotation matrix above
    Eigen::Vector3f eulerAnglesAc;
    eulerAnglesFromRotationZXY(rotationMatAc,
                               eulerAnglesAc.x(),
                               eulerAnglesAc.y(),
                               eulerAnglesAc.z());

    // Store the resulting Euler angles
    acx = eulerAnglesAc.x();
    acy = eulerAnglesAc.y();
    acz = eulerAnglesAc.z();
}

void BasicLaserOdometry::accumulateRotation(
    Angle cx, Angle cy, Angle cz,
    Angle lx, Angle ly, Angle lz,
    Angle& ox, Angle& oy, Angle& oz)
{
    // Create rotation matrices R_c = Ry(cy) Rx(cx) Rz(cz) and
    // R_l = Ry(ly) Rx(lx) Rz(lz)
    const Eigen::Matrix3f rotationMatC =
        rotationMatrixZXY(cx.rad(), cy.rad(), cz.rad());
    const Eigen::Matrix3f rotationMatL =
        rotationMatrixZXY(lx.rad(), ly.rad(), lz.rad());

    // Compose two rotation matrices
    const Eigen::Matrix3f rotationMatO = rotationMatC * rotationMatL;
    // Get three Euler angles from the rotation matrix above
    Eigen::Vector3f eulerAnglesO;
    eulerAnglesFromRotationZXY(rotationMatO,
                               eulerAnglesO.x(),
                               eulerAnglesO.y(),
                               eulerAnglesO.z());

    // Store the resulting Euler angles
    ox = eulerAnglesO.x();
    oy = eulerAnglesO.y();
    oz = eulerAnglesO.z();
}

void BasicLaserOdometry::updateIMU(
    const pcl::PointCloud<pcl::PointXYZ>& imuTrans)
{
    // This method is called from `LaserOdometry::imuTransHandler()`
    assert(imuTrans.size() == 4);

    // This corresponds to `_imuStart` in BasicScanRegistration
    this->_imuPitchStart = imuTrans.points[0].x;
    this->_imuYawStart = imuTrans.points[0].y;
    this->_imuRollStart = imuTrans.points[0].z;

    // This corresponds to `_imuCur` in BasicScanRegistration
    this->_imuPitchEnd = imuTrans.points[1].x;
    this->_imuYawEnd = imuTrans.points[1].y;
    this->_imuRollEnd = imuTrans.points[1].z;

    // This corresponds to `imuShiftFromStart` in BasicScanRegistration
    this->_imuShiftFromStart = imuTrans.points[2];
    // This corresponds to `imuVelocityFromStart` in BasicScanRegistration
    this->_imuVeloFromStart = imuTrans.points[3];
}

void BasicLaserOdometry::process()
{
    const ros::Time startTime = ros::Time::now();

    if (!this->_systemInited) {
        // `_lastCornerCloud` includes both less sharp and sharp points
        // (`CORNER_LESS_SHARP` and `CORNER_SHARP`)
        this->_cornerPointsLessSharp.swap(this->_lastCornerCloud);
        // `_lastSurfaceCloud` includes both less flat and flat points
        // (`SURFACE_LESS_FLAT` and `SURFACE_FLAT`)
        this->_surfPointsLessFlat.swap(this->_lastSurfaceCloud);

        this->_lastCornerKDTree.setInputCloud(this->_lastCornerCloud);
        this->_lastSurfaceKDTree.setInputCloud(this->_lastSurfaceCloud);

        this->_transformSum.rot_x += this->_imuPitchStart;
        this->_transformSum.rot_z += this->_imuRollStart;

        this->_systemInited = true;

        // Update the metric
        const ros::Time endTime = ros::Time::now();
        this->_metricsMsg.process_time = endTime - startTime;

        return;
    }

    this->_frameCount++;

    if (this->_frameCount % 10 == 0)
        ROS_INFO("Processing the frame: %ld", this->_frameCount);

    // Initialize the transform between the odometry poses
    // `_imuVeloFromStart * _scanPeriod` could be multiplied by 0.5
    // since `_imuVeloFromStart` is the multiply of the acceleration and
    // `_scanPeriod`
    this->_transform.pos -= this->_imuVeloFromStart * this->_scanPeriod;

    // Collect the metrics
    this->_metricsMsg.num_of_query_sharp_points =
        this->_cornerPointsSharp->size();
    this->_metricsMsg.num_of_reference_sharp_points =
        this->_lastCornerCloud->size();
    this->_metricsMsg.num_of_query_flat_points =
        this->_surfPointsFlat->size();
    this->_metricsMsg.num_of_reference_flat_points =
        this->_lastSurfaceCloud->size();

    // Perform the Gauss-Newton optimization to update the pose transformation
    // if the previous point cloud is sufficiently large
    if (this->_lastCornerCloud->points.size() > 10 &&
        this->_lastSurfaceCloud->points.size() > 100)
        this->performOptimization();
    else
        ROS_WARN("Pose is not optimized in LaserOdometry node "
                 "since the point cloud is too small: "
                 "_lastCornerCloud->size(): %zu, "
                 "_lastSurfaceCloud->size(): %zu",
                 this->_lastCornerCloud->size(),
                 this->_lastSurfaceCloud->size());

    // `_transformSum` is the transformation from the world coordinate frame
    // to the previous odometry frame
    // `-_transform` is the transformation from the previous odometry frame to
    // the current odometry frame, meaning that the transpose of
    // `_transform.rot` is the rotation from the previous odometry frame to the
    // current odometry frame and `-_transform.pos` is the translation from the
    // previous odometry frame to the current odometry frame
    Angle rx;
    Angle ry;
    Angle rz;
    this->accumulateRotation(this->_transformSum.rot_x,
                             this->_transformSum.rot_y,
                             this->_transformSum.rot_z,
                             -this->_transform.rot_x,
                             -this->_transform.rot_y.rad() * 1.05f,
                             -this->_transform.rot_z,
                             rx, ry, rz);

    Vector3 v {
        this->_transform.pos.x() - this->_imuShiftFromStart.x(),
        this->_transform.pos.y() - this->_imuShiftFromStart.y(),
        this->_transform.pos.z() * 1.05f - this->_imuShiftFromStart.z() };
    rotateZXY(v, rz, rx, ry);
    const Vector3 trans = this->_transformSum.pos - v;

    // Update the rotation using IMU states at the beginning of the current
    // sweep and at the current scan to consider the nonlinear motion
    this->pluginIMURotation(
        rx, ry, rz,
        this->_imuPitchStart, this->_imuYawStart, this->_imuRollStart,
        this->_imuPitchEnd, this->_imuYawEnd, this->_imuRollEnd,
        rx, ry, rz);

    this->_transformSum.rot_x = rx;
    this->_transformSum.rot_y = ry;
    this->_transformSum.rot_z = rz;
    this->_transformSum.pos = trans;

    this->transformToEnd(this->_cornerPointsLessSharp);
    this->transformToEnd(this->_surfPointsLessFlat);

    // Update the corner and planar points for the next scan
    this->_cornerPointsLessSharp.swap(this->_lastCornerCloud);
    this->_surfPointsLessFlat.swap(this->_lastSurfaceCloud);

    if (this->_lastCornerCloud->points.size() > 10 &&
        this->_lastSurfaceCloud->points.size() > 100) {
        this->_lastCornerKDTree.setInputCloud(this->_lastCornerCloud);
        this->_lastSurfaceKDTree.setInputCloud(this->_lastSurfaceCloud);
    }

    // Update the metric
    const ros::Time endTime = ros::Time::now();
    this->_metricsMsg.process_time = endTime - startTime;
}

// Perform the Gauss-Newton optimization and update the pose transformation
void BasicLaserOdometry::performOptimization()
{
    removeNaNFromPointCloud<pcl::PointXYZI>(this->_cornerPointsSharp);

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    // Perform iterations of the Gauss-Newton method
    for (std::size_t iter = 0; iter < this->_maxIterations; ++iter) {
        const ros::Time iterationStartTime = ros::Time::now();

        this->_laserCloudOri->clear();
        this->_coeffSel->clear();

        // Compute the distances and coefficients from the point-to-edge
        // correspondences
        this->computeCornerDistances(iter);
        // Compute the distances and coefficients from the point-to-plane
        // correspondences
        this->computePlaneDistances(iter);

        // Collect the metric
        this->_metricsMsg.num_of_correspondences.push_back(
            this->_laserCloudOri->size());

        // If the number of selected points is less than 10, move to the next
        // iteration and do not perform the following optimization
        const int pointSelNum = this->_laserCloudOri->points.size();

        if (pointSelNum < 10) {
            // Collect the metric
            const ros::Time iterationEndTime = ros::Time::now();
            this->_metricsMsg.optimization_iteration_times.push_back(
                iterationEndTime - iterationStartTime);
            continue;
        }

        // `matA` is the Jacobian matrix in Equation (12)
        Eigen::Matrix<float, Eigen::Dynamic, 6> matA(pointSelNum, 6);
        Eigen::Matrix<float, 6, Eigen::Dynamic> matAt(6, pointSelNum);
        // `matB` is the distance vector (-d) in Equation (12)
        Eigen::VectorXf vecB(pointSelNum);

        for (int i = 0; i < pointSelNum; ++i) {
            const auto& pointOri = this->_laserCloudOri->points[i];
            const auto& coeff = this->_coeffSel->points[i];

            const float s = 1.0;

            const float posX = s * this->_transform.pos.x();
            const float posY = s * this->_transform.pos.y();
            const float posZ = s * this->_transform.pos.z();
            const float rotX = s * this->_transform.rot_x.rad();
            const float rotY = s * this->_transform.rot_y.rad();
            const float rotZ = s * this->_transform.rot_z.rad();

            // Create a rotation matrix and a translation vector from the
            // current `_transform`, note that Euler angles are scaled and
            // their signs are flipped
            const Eigen::Matrix3f rotationMatTrans =
                rotationMatrixYXZT(rotX, rotY, rotZ);
            const Eigen::Vector3f vecTrans { posX, posY, posZ };

            // Compute partial derivatives of the rotation matrix
            const Eigen::Matrix3f rotationMatParX =
                s * partialXFromRotationYXZT(rotX, rotY, rotZ);
            const Eigen::Matrix3f rotationMatParY =
                s * partialYFromRotationYXZT(rotX, rotY, rotZ);
            const Eigen::Matrix3f rotationMatParZ =
                s * partialZFromRotationYXZT(rotX, rotY, rotZ);

            // Create a position vector of a point
            pcl::Vector3fMapConst vecPoint = pointOri.getVector3fMap();
            // Create an intermediate vector
            const Eigen::Vector3f vecPointTrans = vecPoint - vecTrans;

            // Create a coefficient vector
            pcl::Vector3fMapConst vecCoeff = coeff.getVector3fMap();
            // Compute a partial derivative of the point-to-edge or
            // point-to-plane distance with respect to the translation
            const Eigen::Vector3f vecGradTrans =
                -s * rotationMatTrans.transpose() * vecCoeff;
            // Compute a partial derivative of the point-to-edge or
            // point-to-plane distance with respect to the rotation
            const Eigen::Vector3f vecGradRot {
                (rotationMatParX * vecPointTrans).transpose() * vecCoeff,
                (rotationMatParY * vecPointTrans).transpose() * vecCoeff,
                (rotationMatParZ * vecPointTrans).transpose() * vecCoeff };

            matA.block<1, 3>(i, 0) = vecGradRot;
            matA.block<1, 3>(i, 3) = vecGradTrans;

            // Point-to-edge or point-to-plane distance is stored in the
            // intensity field in the coefficient
            // Reverse the sign of the residual to follow Gauss-Newton method
            vecB(i, 0) = -this->_residualScale * coeff.intensity;
        }

        matAt = matA.transpose();
        // `matAtA` is the Hessian matrix (J^T J) in Equation (12)
        // Note that the damping factor is not used in this implementation
        const Eigen::Matrix<float, 6, 6> matAtA = matAt * matA;
        // `matAtB` is the residual vector (-J^T d) in Equation (12)
        const Eigen::VectorXf vecAtB = matAt * vecB;

        // Compute the increment to the current transformation
        // `matX` is the solution to `matAtA` * `matX` = `matAtB`,
        // which is used for updating the current transformation
        Eigen::Matrix<float, 6, 1> vecX =
            matAtA.colPivHouseholderQr().solve(vecAtB);

        // Check the occurrence of the degeneration
        if (iter == 0)
            isDegenerate = this->checkDegeneration(matAtA, matP);

        // Do not update the transformation along the degenerate direction
        if (isDegenerate) {
            Eigen::Matrix<float, 6, 1> matX2(vecX);
            vecX = matP * matX2;
        }

        // Update the transformation (rotation and translation)
        this->_transform.rot_x = this->_transform.rot_x.rad() + vecX(0, 0);
        this->_transform.rot_y = this->_transform.rot_y.rad() + vecX(1, 0);
        this->_transform.rot_z = this->_transform.rot_z.rad() + vecX(2, 0);
        this->_transform.pos.x() += vecX(3, 0);
        this->_transform.pos.y() += vecX(4, 0);
        this->_transform.pos.z() += vecX(5, 0);

        // Reset the transformation if values are invalid (NaN or infinity)
        if (!std::isfinite(this->_transform.rot_x.rad()))
            this->_transform.rot_x = Angle();
        if (!std::isfinite(this->_transform.rot_y.rad()))
            this->_transform.rot_y = Angle();
        if (!std::isfinite(this->_transform.rot_z.rad()))
            this->_transform.rot_z = Angle();

        if (!std::isfinite(this->_transform.pos.x()))
            this->_transform.pos.x() = 0.0f;
        if (!std::isfinite(this->_transform.pos.y()))
            this->_transform.pos.y() = 0.0f;
        if (!std::isfinite(this->_transform.pos.z()))
            this->_transform.pos.z() = 0.0f;

        // Compute the increment in degrees and centimeters
        const float deltaR = std::sqrt(
            std::pow(rad2deg(vecX(0, 0)), 2)
            + std::pow(rad2deg(vecX(1, 0)), 2)
            + std::pow(rad2deg(vecX(2, 0)), 2));
        const float deltaT = std::sqrt(
            std::pow(vecX(3, 0) * 100, 2)
            + std::pow(vecX(4, 0) * 100, 2)
            + std::pow(vecX(5, 0) * 100, 2));

        // Collect the metric
        const ros::Time iterationEndTime = ros::Time::now();
        this->_metricsMsg.optimization_iteration_times.push_back(
            iterationEndTime - iterationStartTime);

        // Terminate the Gauss-Newton method if the increment is small
        if (deltaR < this->_deltaRAbort && deltaT < this->_deltaTAbort)
            break;
    }

    // Collect the metrics
    this->_metricsMsg.optimization_time = std::accumulate(
        this->_metricsMsg.optimization_iteration_times.begin(),
        this->_metricsMsg.optimization_iteration_times.end(),
        ros::Duration(0.0));
    this->_metricsMsg.num_of_iterations =
        this->_metricsMsg.optimization_iteration_times.size();
}

// Check the occurrence of the degeneration
bool BasicLaserOdometry::checkDegeneration(
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

    Eigen::Matrix<float, 1, 6> matE = eigenSolver.eigenvalues().real();
    Eigen::Matrix<float, 6, 6> matV = eigenSolver.eigenvectors().real();
    Eigen::Matrix<float, 6, 6> matV2 = matV;

    bool isDegenerate = false;
    const float eigenThreshold[6] = {
        this->_eigenThresholdRot, this->_eigenThresholdRot,
        this->_eigenThresholdRot, this->_eigenThresholdTrans,
        this->_eigenThresholdTrans, this->_eigenThresholdTrans };

    // Eigenvalues are sorted in the increasing order
    // Detect the occurrence of the degeneration if one of the eigenvalues is
    // less than `_eigenThresholdTrans` or `_eigenThresholdRot`
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
void BasicLaserOdometry::computeCornerDistances(int iterCount)
{
    if (iterCount % 5 == 0) {
        this->findCornerCorrespondence();
    } else {
        // Collect the metrics (for convenience)
        this->_metricsMsg.corner_correspondence_times.push_back(
            ros::Duration(0.0));
        this->_metricsMsg.num_of_corner_correspondences.push_back(
            this->_metricsMsg.num_of_corner_correspondences.back());
    }

    const ros::Time startTime = ros::Time::now();

    const std::size_t cornerPointsSharpNum =
        this->_cornerPointsSharp->points.size();

    // For each corner point in the current scan, find the closest neighbor
    // point in the last scan which is reprojected to the beginning of the
    // current sweep (i.e., current scan, since each sweep contains only one
    // scan in this implementation)
    for (int i = 0; i < cornerPointsSharpNum; ++i) {
        // Reproject the corner point in the current scan to the beginning
        // of the current sweep (point `i` in the paper)
        const pcl::PointXYZI pointSel =
            this->transformToStart(this->_cornerPointsSharp->points[i]);

        if (this->_pointSearchCornerInd2[i] < 0)
            continue;

        // `ind1[i]` should be valid if `ind2[i]` is valid
        const pcl::PointXYZI tripod1 =
            this->_lastCornerCloud->points[this->_pointSearchCornerInd1[i]];
        const pcl::PointXYZI tripod2 =
            this->_lastCornerCloud->points[this->_pointSearchCornerInd2[i]];

        // `pointSel`, `tripod1`, and `tripod2` correspond to
        // X^L_(k + 1, i), X^L_(k, j), and X^L_(k, l) in Equation (2)
        // which are reprojected to the beginning of the current sweep
        // (i.e., timestamp t_(k + 1) in the paper)
        const Eigen::Vector3f vecI { pointSel.x, pointSel.y, pointSel.z };
        const Eigen::Vector3f vecJ { tripod1.x, tripod1.y, tripod1.z };
        const Eigen::Vector3f vecL { tripod2.x, tripod2.y, tripod2.z };

        const Eigen::Vector3f vecIJ = vecI - vecJ;
        const Eigen::Vector3f vecIL = vecI - vecL;
        const Eigen::Vector3f vecJL = vecJ - vecL;
        const Eigen::Vector3f vecCross = vecIJ.cross(vecIL);

        // Compute the numerator of the Equation (2)
        const float a012 = vecCross.norm();
        // Compute the denominator of the Equation (2)
        const float l12 = vecJL.norm();

        // Compute the normal vector (la, lb, lc) from the projection of the
        // point `i` on the edge line between points `j` and `l` and the
        // point `i`
        const Eigen::Vector3f vecNormal = vecJL.cross(vecCross) / a012 / l12;
        // Compute the d_e using the Equation (2), which is the point-to-edge
        // distance between the point `i` and the edge line between points
        // `j` and `l` in the previous scan
        const float ld2 = a012 / l12;

        // Compute the projection of the point `i` on the edge line
        // between points `j` and `l` using the above normal vector
        const Eigen::Vector3f vecProj = vecI - vecNormal * ld2;

        // Assign smaller weights for the points with larger
        // point-to-edge distances and zero weights for outliers
        // with distances larger than the threshold (Section V.D)
        const float s = iterCount < 5 ? 1.0f :
                        (1.0f - this->_weightDecayCorner * std::fabs(ld2));

        if (s <= this->_weightThresholdCorner || ld2 == 0.0f)
            continue;

        // Store the coefficient vector and the original point `i`
        // that is not reprojected to the beginning of the current sweep
        pcl::PointXYZI coeff;
        coeff.x = s * vecNormal.x();
        coeff.y = s * vecNormal.y();
        coeff.z = s * vecNormal.z();
        coeff.intensity = s * ld2;

        this->_laserCloudOri->push_back(this->_cornerPointsSharp->points[i]);
        this->_coeffSel->push_back(coeff);
    }

    // Collect the metric
    const ros::Time endTime = ros::Time::now();
    this->_metricsMsg.corner_coefficient_times.push_back(
        endTime - startTime);
}

// Find point-to-edge correspondences from the corner point cloud
void BasicLaserOdometry::findCornerCorrespondence()
{
    const ros::Time startTime = ros::Time::now();

    std::size_t numOfValidCorrespondences = 0;

    const std::size_t cornerPointsSharpNum =
        this->_cornerPointsSharp->points.size();
    this->_pointSearchCornerInd1.resize(cornerPointsSharpNum);
    this->_pointSearchCornerInd2.resize(cornerPointsSharpNum);

    removeNaNFromPointCloud<pcl::PointXYZI>(this->_lastCornerCloud);

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    pointSearchInd.resize(1);
    pointSearchSqDis.resize(1);

    // For each corner point in the current scan, find the closest neighbor
    // point in the last scan which is reprojected to the beginning of the
    // current sweep (i.e., current scan, since each sweep contains only one
    // scan in this implementation)
    for (int i = 0; i < cornerPointsSharpNum; ++i) {
        // Reproject the corner point in the current scan to the beginning
        // of the current sweep (point `i` in the paper)
        const pcl::PointXYZI pointSel =
            this->transformToStart(this->_cornerPointsSharp->points[i]);

        // Find the closest point in the last scan for `pointSel`,
        // which is the point `j` in the paper
        this->_lastCornerKDTree.nearestKSearch(
            pointSel, 1, pointSearchInd, pointSearchSqDis);

        // If the distance between the corner point in the current scan
        // (point `i` in the paper) and its closest point in the last
        // scan (point `j` in the paper) is larger than 5 meters, then the
        // correspondence for the current corner point `i` is not found
        if (pointSearchSqDis[0] >= this->_sqDistThresholdCorner) {
            this->_pointSearchCornerInd1[i] = -1;
            this->_pointSearchCornerInd2[i] = -1;
            continue;
        }

        // Get the scan ID of the closest point in the last scan
        // (scan ID corresponds to the scan ring in Velodyne LiDAR,
        // and points in the same ring have the same vertical angle)
        const int closestPointInd = pointSearchInd[0];
        const auto& closestPoint =
            this->_lastCornerCloud->points[closestPointInd];
        const int closestPointScan =
            static_cast<int>(closestPoint.intensity);

        // Find the closest point of the corner point `pointSel`
        // in the two consecutive scans to the scan of point `j`
        // which is the point `l` in the paper
        int minPointInd2 = -1;
        float minPointSqDis2 = this->_sqDistThresholdCorner;

        // The below should be `j < lastCornerCloudSize` instead of
        // `j < cornerPointsSharpNum`
        const int lastCornerCloudSize = this->_lastCornerCloud->size();
        // for (int j = closestPointInd + 1; j < cornerPointsSharpNum; ++j) {
        for (int j = closestPointInd + 1; j < lastCornerCloudSize; ++j) {
            const auto& scanPoint = this->_lastCornerCloud->points[j];
            const int scanId = static_cast<int>(scanPoint.intensity);

            // If the difference of the scan ID (i.e., difference of
            // the vertical angle) is larger than 2.5, terminate
            if (scanId > closestPointScan + 2.5f)
                break;
            // Skip the points in the same scan as point `j`
            if (scanId <= closestPointScan)
                continue;

            // `minPointInd2` is the index of the point `l` and
            // `minPointSqDis2` is the distance between the current
            // corner point `pointSel` (point `i`) and point `l`
            const float pointSqDis = calcSquaredDiff(scanPoint, pointSel);

            if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
            }
        }

        for (int j = closestPointInd - 1; j >= 0; --j) {
            const auto& scanPoint = this->_lastCornerCloud->points[j];
            const int scanId = static_cast<int>(scanPoint.intensity);

            if (scanId < closestPointScan - 2.5f)
                break;
            if (scanId >= closestPointScan)
                continue;

            const float pointSqDis = calcSquaredDiff(scanPoint, pointSel);

            if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
            }
        }

        // Point `i` is the current corner point in the scan (`pointSel`
        // stored in `_cornerPointsSharp`) and points `j` and `l` are its
        // closest points in the corner points of the last scan
        // (`closestPointInd` and `minPointInd2` in `_lastCornerCloud`)
        this->_pointSearchCornerInd1[i] = closestPointInd;
        this->_pointSearchCornerInd2[i] = minPointInd2;

        if (minPointInd2 != -1)
            ++numOfValidCorrespondences;
    }

    // Collect the metrics
    const ros::Time endTime = ros::Time::now();
    this->_metricsMsg.corner_correspondence_times.push_back(
        endTime - startTime);
    this->_metricsMsg.num_of_corner_correspondences.push_back(
        numOfValidCorrespondences);
}

// Compute the distances and coefficients from the point-to-plane
// correspondences
void BasicLaserOdometry::computePlaneDistances(int iterCount)
{
    if (iterCount % 5 == 0) {
        this->findPlaneCorrespondence();
    } else {
        // Collect the metrics (for convenience)
        this->_metricsMsg.plane_correspondence_times.push_back(
            ros::Duration(0.0));
        this->_metricsMsg.num_of_plane_correspondences.push_back(
            this->_metricsMsg.num_of_plane_correspondences.back());
    }

    const ros::Time startTime = ros::Time::now();

    const std::size_t surfPointsFlatNum =
        this->_surfPointsFlat->points.size();

    // For each planar point in the current scan (stored in `_surfPointsFlat`),
    // find the closest neighbor point in the last scan (stored in
    // `_lastSurfaceCloud`) which is reprojected to the beginning of the
    // current sweep (i.e., timestamp of the current scan)
    for (int i = 0; i < surfPointsFlatNum; ++i) {
        // Reproject the planar point in the current scan to the beginning
        // of the current sweep (point `i` in the paper)
        const pcl::PointXYZI pointSel =
            this->transformToStart(this->_surfPointsFlat->points[i]);

        if (this->_pointSearchSurfInd2[i] < 0 ||
            this->_pointSearchSurfInd3[i] < 0)
            continue;

        // `ind1[i]` is valid if both `ind2[i]` and `ind3[i]` are valid
        // `tripod1`, `tripod2`, and `tripod3` correspond to the
        // points `j`, `l`, and `m`
        const pcl::PointXYZI tripod1 =
            this->_lastSurfaceCloud->points[this->_pointSearchSurfInd1[i]];
        const pcl::PointXYZI tripod2 =
            this->_lastSurfaceCloud->points[this->_pointSearchSurfInd2[i]];
        const pcl::PointXYZI tripod3 =
            this->_lastSurfaceCloud->points[this->_pointSearchSurfInd3[i]];

        const Eigen::Vector3f vecI { pointSel.x, pointSel.y, pointSel.z };
        const Eigen::Vector3f vecJ { tripod1.x, tripod1.y, tripod1.z };
        const Eigen::Vector3f vecL { tripod2.x, tripod2.y, tripod2.z };
        const Eigen::Vector3f vecM { tripod3.x, tripod3.y, tripod3.z };

        const Eigen::Vector3f vecIJ = vecI - vecJ;
        const Eigen::Vector3f vecJL = vecJ - vecL;
        const Eigen::Vector3f vecJM = vecJ - vecM;

        // Compute the vector (pa, pb, pc) that is perpendicular to the
        // plane defined by points `j`, `l`, and `m`, which is written as
        // (X^L_(k, j) - X^L_(k, l)) * (X^L_(k, j) - X^L_(k, m))
        const Eigen::Vector3f vecCross = vecJL.cross(vecJM);
        // Compute the denominator of the Equation (3)
        const float ps = vecCross.norm();
        // Compute the normal vector
        const Eigen::Vector3f vecNormal = vecCross / ps;

        // Compute the d_h using the Equation (3), which is the point-to-plane
        // distance between the point `i` and the plane defined by three points
        // `j`, `l`, and `m`
        // Note that the distance below could be negative
        const float pd2 = vecIJ.dot(vecNormal);

        // Compute the projection of the point `i` on the plane defined by
        // points `j`, `l`, and `m` using the normal vector
        // If the normal vector (pa, pb, pc) and the vector between `i` and `j`
        // point to the same direction, `pd2` is positive and `pointSel` is
        // moved to the opposite direction of the normal vector; otherwise,
        // two vectors point to the opposite direction and `pd2` is negative,
        // which means that `pointSel` is moved to the direction of the normal
        // vector and produces the correct result
        const Eigen::Vector3f vecProj = vecI - vecNormal * pd2;

        // Assign smaller weights for the points with larger
        // point-to-plane distances and zero weights for outliers
        // with distances larger than the threshold (Section V.D)
        const float s = iterCount < 5 ? 1.0f :
            (1.0f - this->_weightDecaySurface * std::fabs(pd2)
            / std::sqrt(calcPointDistance(pointSel)));

        if (s <= this->_weightThresholdSurface || pd2 == 0.0f)
            continue;

        // Store the coefficient vector and the original point `i`
        // that is not reprojected to the beginning of the current sweep
        pcl::PointXYZI coeff;
        coeff.x = s * vecNormal.x();
        coeff.y = s * vecNormal.y();
        coeff.z = s * vecNormal.z();
        coeff.intensity = s * pd2;

        this->_laserCloudOri->push_back(this->_surfPointsFlat->points[i]);
        this->_coeffSel->push_back(coeff);
    }

    // Collect the metric
    const ros::Time endTime = ros::Time::now();
    this->_metricsMsg.plane_coefficient_times.push_back(
        endTime - startTime);
}

// Find point-to-plane correspondences from the planar point cloud
void BasicLaserOdometry::findPlaneCorrespondence()
{
    const ros::Time startTime = ros::Time::now();

    std::size_t numOfValidCorrespondences = 0;

    const std::size_t surfPointsFlatNum =
        this->_surfPointsFlat->points.size();
    this->_pointSearchSurfInd1.resize(surfPointsFlatNum);
    this->_pointSearchSurfInd2.resize(surfPointsFlatNum);
    this->_pointSearchSurfInd3.resize(surfPointsFlatNum);

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    pointSearchInd.resize(1);
    pointSearchSqDis.resize(1);

    // For each planar point in the current scan (stored in `_surfPointsFlat`),
    // find the closest neighbor point in the last scan (stored in
    // `_lastSurfaceCloud`) which is reprojected to the beginning of the
    // current sweep (i.e., timestamp of the current scan)
    for (int i = 0; i < surfPointsFlatNum; ++i) {
        // Reproject the planar point in the current scan to the beginning
        // of the current sweep (point `i` in the paper)
        const pcl::PointXYZI pointSel =
            this->transformToStart(this->_surfPointsFlat->points[i]);

        // Find the closest point in the last scan for `pointSel`,
        // which is the point `j` in the paper
        this->_lastSurfaceKDTree.nearestKSearch(
            pointSel, 1, pointSearchInd, pointSearchSqDis);

        // If the distance between the planar point in the current scan
        // (point `i` in the paper) and its closest point in the last
        // scan (point `j` in the paper) is larger than 5 meters, then the
        // correspondence for the current planar point `i` is not found
        if (pointSearchSqDis[0] >= this->_sqDistThresholdSurface) {
            this->_pointSearchSurfInd1[i] = -1;
            this->_pointSearchSurfInd2[i] = -1;
            this->_pointSearchSurfInd3[i] = -1;
            continue;
        }

        // Get the scan ID of the closest point in the last scan (point `j`)
        const int closestPointInd = pointSearchInd[0];
        const auto& closestPoint =
            this->_lastSurfaceCloud->points[closestPointInd];
        const int closestPointScan =
            static_cast<int>(closestPoint.intensity);

        // Find two points `l` and `m` from the last scan as the
        // closest neighbor points of `i`, one is in the same scan
        // as point `j`, and the other is in the two consecutive
        // scans to the scan of point `j`
        int minPointInd2 = -1;
        int minPointInd3 = -1;
        float minPointSqDis2 = this->_sqDistThresholdSurface;
        float minPointSqDis3 = this->_sqDistThresholdSurface;

        // The below should be `j < _lastSurfaceCloud` instead of
        // `j < surfPointsFlatNum`
        const int lastSurfaceCloudSize = this->_lastSurfaceCloud->size();
        // for (int j = closestPointInd + 1; j < surfPointsFlatNum; ++j) {
        for (int j = closestPointInd + 1; j < lastSurfaceCloudSize; ++j) {
            const auto& scanPoint = this->_lastSurfaceCloud->points[j];
            const int scanId = static_cast<int>(scanPoint.intensity);

            if (scanId > closestPointScan + 2.5f)
                break;

            const float pointSqDis = calcSquaredDiff(scanPoint, pointSel);

            if (scanId <= closestPointScan) {
                // Update the index of the point `l` in the same scan as
                // point `j`, and the distance to the current planar
                // point `pointSel` (point `i`)
                if (pointSqDis < minPointSqDis2) {
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
            } else {
                // Update the index of the point `m` in the two consecutive
                // scans to the scan of point `j`, and the distance to the
                // current planar point `pointSel`
                if (pointSqDis < minPointSqDis3) {
                    minPointSqDis3 = pointSqDis;
                    minPointInd3 = j;
                }
            }
        }

        for (int j = closestPointInd - 1; j >= 0; --j) {
            const auto& scanPoint = this->_lastSurfaceCloud->points[j];
            const int scanId = static_cast<int>(scanPoint.intensity);

            if (scanId < closestPointScan - 2.5f)
                break;

            const float pointSqDis = calcSquaredDiff(scanPoint, pointSel);

            if (scanId >= closestPointScan) {
                if (pointSqDis < minPointSqDis2) {
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
            } else {
                if (pointSqDis < minPointSqDis3) {
                    minPointSqDis3 = pointSqDis;
                    minPointInd3 = j;
                }
            }
        }

        this->_pointSearchSurfInd1[i] = closestPointInd;
        this->_pointSearchSurfInd2[i] = minPointInd2;
        this->_pointSearchSurfInd3[i] = minPointInd3;

        if (minPointInd2 != -1 && minPointInd3 != -1)
            ++numOfValidCorrespondences;
    }

    // Collect the metrics
    const ros::Time endTime = ros::Time::now();
    this->_metricsMsg.plane_correspondence_times.push_back(
        endTime - startTime);
    this->_metricsMsg.num_of_plane_correspondences.push_back(
        numOfValidCorrespondences);
}

// Clear the metrics message
void BasicLaserOdometry::clearMetricsMsg()
{
    this->_metricsMsg.stamp = ros::Time(0.0);

    this->_metricsMsg.point_cloud_stamp = ros::Time(0.0);
    this->_metricsMsg.num_of_full_res_points = 0;
    this->_metricsMsg.num_of_less_sharp_points = 0;
    this->_metricsMsg.num_of_less_flat_points = 0;
    this->_metricsMsg.num_of_dropped_point_clouds = 0;

    this->_metricsMsg.process_time = ros::Duration(0.0);
    this->_metricsMsg.num_of_query_sharp_points = 0;
    this->_metricsMsg.num_of_reference_sharp_points = 0;
    this->_metricsMsg.num_of_query_flat_points = 0;
    this->_metricsMsg.num_of_reference_flat_points = 0;

    this->_metricsMsg.optimization_time = ros::Duration(0.0);
    this->_metricsMsg.num_of_iterations = 0;
    this->_metricsMsg.optimization_iteration_times.clear();
    this->_metricsMsg.num_of_correspondences.clear();

    this->_metricsMsg.corner_coefficient_times.clear();
    this->_metricsMsg.corner_correspondence_times.clear();
    this->_metricsMsg.num_of_corner_correspondences.clear();

    this->_metricsMsg.plane_coefficient_times.clear();
    this->_metricsMsg.plane_correspondence_times.clear();
    this->_metricsMsg.num_of_plane_correspondences.clear();
}

} // namespace loam
