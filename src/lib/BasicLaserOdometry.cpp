
// BasicLaserOdometry.cpp

#include "loam_velodyne/BasicLaserOdometry.h"
#include "math_utils.h"

#include <pcl/filters/filter.h>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace loam {

using std::sin;
using std::cos;
using std::asin;
using std::atan2;
using std::sqrt;
using std::fabs;
using std::pow;

BasicLaserOdometry::BasicLaserOdometry(
    float scanPeriod, std::size_t maxIterations) :
    _scanPeriod(scanPeriod),
    _systemInited(false),
    _frameCount(0),
    _maxIterations(maxIterations),
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

void BasicLaserOdometry::transformToStart(
    const pcl::PointXYZI& pi, pcl::PointXYZI& po)
{
    const float relTime = pi.intensity - static_cast<int>(pi.intensity);
    const float s = (1.0f / this->_scanPeriod) * relTime;

    po.x = pi.x - s * this->_transform.pos.x();
    po.y = pi.y - s * this->_transform.pos.y();
    po.z = pi.z - s * this->_transform.pos.z();
    po.intensity = pi.intensity;

    Angle rx = -s * this->_transform.rot_x.rad();
    Angle ry = -s * this->_transform.rot_y.rad();
    Angle rz = -s * this->_transform.rot_z.rad();
    rotateZXY(po, rz, rx, ry);
}

std::size_t BasicLaserOdometry::transformToEnd(
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
    const std::size_t cloudSize = cloud->points.size();

    for (std::size_t i = 0; i < cloudSize; ++i) {
        pcl::PointXYZI& point = cloud->points[i];

        // Transform to the start of the sweep
        float relTime = point.intensity - static_cast<int>(point.intensity);
        float s = (1.0f / this->_scanPeriod) * relTime;

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
   float sbcx = bcx.sin();
   float cbcx = bcx.cos();
   float sbcy = bcy.sin();
   float cbcy = bcy.cos();
   float sbcz = bcz.sin();
   float cbcz = bcz.cos();

   float sblx = blx.sin();
   float cblx = blx.cos();
   float sbly = bly.sin();
   float cbly = bly.cos();
   float sblz = blz.sin();
   float cblz = blz.cos();

   float salx = alx.sin();
   float calx = alx.cos();
   float saly = aly.sin();
   float caly = aly.cos();
   float salz = alz.sin();
   float calz = alz.cos();

   float srx = -sbcx * (salx*sblx + calx * caly*cblx*cbly + calx * cblx*saly*sbly)
      - cbcx * cbcz*(calx*saly*(cbly*sblz - cblz * sblx*sbly)
                     - calx * caly*(sbly*sblz + cbly * cblz*sblx) + cblx * cblz*salx)
      - cbcx * sbcz*(calx*caly*(cblz*sbly - cbly * sblx*sblz)
                     - calx * saly*(cbly*cblz + sblx * sbly*sblz) + cblx * salx*sblz);
   acx = -asin(srx);

   float srycrx = (cbcy*sbcz - cbcz * sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz * sblx*sbly)
                                                  - calx * caly*(sbly*sblz + cbly * cblz*sblx) + cblx * cblz*salx)
      - (cbcy*cbcz + sbcx * sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly * sblx*sblz)
                                        - calx * saly*(cbly*cblz + sblx * sbly*sblz) + cblx * salx*sblz)
      + cbcx * sbcy*(salx*sblx + calx * caly*cblx*cbly + calx * cblx*saly*sbly);
   float crycrx = (cbcz*sbcy - cbcy * sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly * sblx*sblz)
                                                  - calx * saly*(cbly*cblz + sblx * sbly*sblz) + cblx * salx*sblz)
      - (sbcy*sbcz + cbcy * cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz * sblx*sbly)
                                        - calx * caly*(sbly*sblz + cbly * cblz*sblx) + cblx * cblz*salx)
      + cbcx * cbcy*(salx*sblx + calx * caly*cblx*cbly + calx * cblx*saly*sbly);
   acy = atan2(srycrx / acx.cos(), crycrx / acx.cos());

   float srzcrx = sbcx * (cblx*cbly*(calz*saly - caly * salx*salz) - cblx * sbly*(caly*calz + salx * saly*salz) + calx * salz*sblx)
      - cbcx * cbcz*((caly*calz + salx * saly*salz)*(cbly*sblz - cblz * sblx*sbly)
                     + (calz*saly - caly * salx*salz)*(sbly*sblz + cbly * cblz*sblx)
                     - calx * cblx*cblz*salz)
      + cbcx * sbcz*((caly*calz + salx * saly*salz)*(cbly*cblz + sblx * sbly*sblz)
                     + (calz*saly - caly * salx*salz)*(cblz*sbly - cbly * sblx*sblz)
                     + calx * cblx*salz*sblz);
   float crzcrx = sbcx * (cblx*sbly*(caly*salz - calz * salx*saly) - cblx * cbly*(saly*salz + caly * calz*salx) + calx * calz*sblx)
      + cbcx * cbcz*((saly*salz + caly * calz*salx)*(sbly*sblz + cbly * cblz*sblx)
                     + (caly*salz - calz * salx*saly)*(cbly*sblz - cblz * sblx*sbly)
                     + calx * calz*cblx*cblz)
      - cbcx * sbcz*((saly*salz + caly * calz*salx)*(cblz*sbly - cbly * sblx*sblz)
                     + (caly*salz - calz * salx*saly)*(cbly*cblz + sblx * sbly*sblz)
                     - calx * calz*cblx*sblz);
   acz = atan2(srzcrx / acx.cos(), crzcrx / acx.cos());
}

void BasicLaserOdometry::accumulateRotation(
    Angle cx, Angle cy, Angle cz,
    Angle lx, Angle ly, Angle lz,
    Angle& ox, Angle& oy, Angle& oz)
{
   float srx = lx.cos()*cx.cos()*ly.sin()*cz.sin()
      - cx.cos()*cz.cos()*lx.sin()
      - lx.cos()*ly.cos()*cx.sin();
   ox = -asin(srx);

   float srycrx = lx.sin()*(cy.cos()*cz.sin() - cz.cos()*cx.sin()*cy.sin())
      + lx.cos()*ly.sin()*(cy.cos()*cz.cos() + cx.sin()*cy.sin()*cz.sin())
      + lx.cos()*ly.cos()*cx.cos()*cy.sin();
   float crycrx = lx.cos()*ly.cos()*cx.cos()*cy.cos()
      - lx.cos()*ly.sin()*(cz.cos()*cy.sin() - cy.cos()*cx.sin()*cz.sin())
      - lx.sin()*(cy.sin()*cz.sin() + cy.cos()*cz.cos()*cx.sin());
   oy = atan2(srycrx / ox.cos(), crycrx / ox.cos());

   float srzcrx = cx.sin()*(lz.cos()*ly.sin() - ly.cos()*lx.sin()*lz.sin())
      + cx.cos()*cz.sin()*(ly.cos()*lz.cos() + lx.sin()*ly.sin()*lz.sin())
      + lx.cos()*cx.cos()*cz.cos()*lz.sin();
   float crzcrx = lx.cos()*lz.cos()*cx.cos()*cz.cos()
      - cx.cos()*cz.sin()*(ly.cos()*lz.sin() - lz.cos()*lx.sin()*ly.sin())
      - cx.sin()*(ly.sin()*lz.sin() + ly.cos()*lz.cos()*lx.sin());
   oz = atan2(srzcrx / ox.cos(), crzcrx / ox.cos());
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

        return;
    }

    this->_frameCount++;

    // Initialize the transform between the odometry poses
    // `_imuVeloFromStart * _scanPeriod` could be multiplied by 0.5
    // since `_imuVeloFromStart` is the multiply of the acceleration and
    // `_scanPeriod`
    this->_transform.pos -= this->_imuVeloFromStart * this->_scanPeriod;

    // Perform the Gauss-Newton optimization to update the pose transformation
    // if the previous point cloud is sufficiently large
    if (this->_lastCornerCloud->points.size() > 10 &&
        this->_lastSurfaceCloud->points.size() > 100)
        this->performOptimization();

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
}

// Perform the Gauss-Newton optimization and update the pose transformation
void BasicLaserOdometry::performOptimization()
{
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(
        *this->_cornerPointsSharp, *this->_cornerPointsSharp, indices);

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    // Perform iterations of the Gauss-Newton method
    for (std::size_t iterCount = 0;
         iterCount < this->_maxIterations; ++iterCount) {
        this->_laserCloudOri->clear();
        this->_coeffSel->clear();

        // Compute the distances and coefficients from the point-to-edge
        // correspondences
        this->computeCornerDistances(iterCount);
        // Compute the distances and coefficients from the point-to-plane
        // correspondences
        this->computePlaneDistances(iterCount);

        // If the number of selected points is less than 10, move to the next
        // iteration and do not perform the following optimization
        const int pointSelNum = this->_laserCloudOri->points.size();

        if (pointSelNum < 10)
            continue;

        // `matA` is the Jacobian matrix in Equation (12)
        Eigen::Matrix<float, Eigen::Dynamic, 6> matA(pointSelNum, 6);
        Eigen::Matrix<float, 6, Eigen::Dynamic> matAt(6, pointSelNum);
        // `matAtA` is the Hessian matrix (J^T J) in Equation (12)
        // Note that the damping factor is not used in this implementation
        Eigen::Matrix<float, 6, 6> matAtA;
        // `matB` is the distance vector (-d) in Equation (12)
        Eigen::VectorXf matB(pointSelNum);
        // `matAtB` is the residual vector (-J^T d) in Equation (12)
        Eigen::Matrix<float, 6, 1> matAtB;
        // `matX` is the solution to `matAtA` * `matX` = `matAtB`,
        // which is used for updating the current transformation
        Eigen::Matrix<float, 6, 1> matX;

        for (int i = 0; i < pointSelNum; ++i) {
            const auto& pointOri = this->_laserCloudOri->points[i];
            const auto& coeff = this->_coeffSel->points[i];

            const float s = 1.0;

            const float srx = std::sin(s * this->_transform.rot_x.rad());
            const float crx = std::cos(s * this->_transform.rot_x.rad());
            const float sry = std::sin(s * this->_transform.rot_y.rad());
            const float cry = std::cos(s * this->_transform.rot_y.rad());
            const float srz = std::sin(s * this->_transform.rot_z.rad());
            const float crz = std::cos(s * this->_transform.rot_z.rad());
            const float tx = s * this->_transform.pos.x();
            const float ty = s * this->_transform.pos.y();
            const float tz = s * this->_transform.pos.z();

            float arx = (-s * crx*sry*srz*pointOri.x + s * crx*crz*sry*pointOri.y + s * srx*sry*pointOri.z
                        + s * tx*crx*sry*srz - s * ty*crx*crz*sry - s * tz*srx*sry) * coeff.x
                        + (s*srx*srz*pointOri.x - s * crz*srx*pointOri.y + s * crx*pointOri.z
                        + s * ty*crz*srx - s * tz*crx - s * tx*srx*srz) * coeff.y
                        + (s*crx*cry*srz*pointOri.x - s * crx*cry*crz*pointOri.y - s * cry*srx*pointOri.z
                        + s * tz*cry*srx + s * ty*crx*cry*crz - s * tx*crx*cry*srz) * coeff.z;

            float ary = ((-s * crz*sry - s * cry*srx*srz)*pointOri.x
                        + (s*cry*crz*srx - s * sry*srz)*pointOri.y - s * crx*cry*pointOri.z
                        + tx * (s*crz*sry + s * cry*srx*srz) + ty * (s*sry*srz - s * cry*crz*srx)
                        + s * tz*crx*cry) * coeff.x
                        + ((s*cry*crz - s * srx*sry*srz)*pointOri.x
                        + (s*cry*srz + s * crz*srx*sry)*pointOri.y - s * crx*sry*pointOri.z
                        + s * tz*crx*sry - ty * (s*cry*srz + s * crz*srx*sry)
                        - tx * (s*cry*crz - s * srx*sry*srz)) * coeff.z;

            float arz = ((-s * cry*srz - s * crz*srx*sry)*pointOri.x + (s*cry*crz - s * srx*sry*srz)*pointOri.y
                        + tx * (s*cry*srz + s * crz*srx*sry) - ty * (s*cry*crz - s * srx*sry*srz)) * coeff.x
                        + (-s * crx*crz*pointOri.x - s * crx*srz*pointOri.y
                        + s * ty*crx*srz + s * tx*crx*crz) * coeff.y
                        + ((s*cry*crz*srx - s * sry*srz)*pointOri.x + (s*crz*sry + s * cry*srx*srz)*pointOri.y
                        + tx * (s*sry*srz - s * cry*crz*srx) - ty * (s*crz*sry + s * cry*srx*srz)) * coeff.z;

            float atx = -s * (cry*crz - srx * sry*srz) * coeff.x + s * crx*srz * coeff.y
                        - s * (crz*sry + cry * srx*srz) * coeff.z;

            float aty = -s * (cry*srz + crz * srx*sry) * coeff.x - s * crx*crz * coeff.y
                        - s * (sry*srz - cry * crz*srx) * coeff.z;

            float atz = s * crx*sry * coeff.x - s * srx * coeff.y - s * crx*cry * coeff.z;

            float d2 = coeff.intensity;

            matA(i, 0) = arx;
            matA(i, 1) = ary;
            matA(i, 2) = arz;
            matA(i, 3) = atx;
            matA(i, 4) = aty;
            matA(i, 5) = atz;
            // Reverse the sign of the residual to follow Gauss-Newton method
            matB(i, 0) = -0.05 * d2;
        }

        matAt = matA.transpose();
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        // Compute the increment to the current transformation
        matX = matAtA.colPivHouseholderQr().solve(matAtB);

        // Check the occurrence of the degeneration
        if (iterCount == 0)
            isDegenerate = this->checkDegeneration(matAtA, matP);

        // Do not update the transformation along the degenerate direction
        if (isDegenerate) {
            Eigen::Matrix<float, 6, 1> matX2(matX);
            matX = matP * matX2;
        }

        // Update the transformation (rotation and translation)
        this->_transform.rot_x = this->_transform.rot_x.rad() + matX(0, 0);
        this->_transform.rot_y = this->_transform.rot_y.rad() + matX(1, 0);
        this->_transform.rot_z = this->_transform.rot_z.rad() + matX(2, 0);
        this->_transform.pos.x() += matX(3, 0);
        this->_transform.pos.y() += matX(4, 0);
        this->_transform.pos.z() += matX(5, 0);

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
    const float eigenThreshold[6] = { 10, 10, 10, 10, 10, 10 };

    // Eigenvalues are sorted in the increasing order
    // Detect the occurrence of the degeneration if one of the eigenvalues is
    // less than 10
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
    if (iterCount % 5 == 0)
        this->findCornerCorrespondence();

    const std::size_t cornerPointsSharpNum =
        this->_cornerPointsSharp->points.size();

    // For each corner point in the current scan, find the closest neighbor
    // point in the last scan which is reprojected to the beginning of the
    // current sweep (i.e., current scan, since each sweep contains only one
    // scan in this implementation)
    for (int i = 0; i < cornerPointsSharpNum; ++i) {
        // Reproject the corner point in the current scan to the beginning
        // of the current sweep (point `i` in the paper)
        pcl::PointXYZI pointSel;
        this->transformToStart(this->_cornerPointsSharp->points[i], pointSel);

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
        const float x0 = pointSel.x;
        const float y0 = pointSel.y;
        const float z0 = pointSel.z;
        const float x1 = tripod1.x;
        const float y1 = tripod1.y;
        const float z1 = tripod1.z;
        const float x2 = tripod2.x;
        const float y2 = tripod2.y;
        const float z2 = tripod2.z;

        // Compute the numerator of the Equation (2)
        const float a012 = std::sqrt(
            ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
            * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
            + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
            * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
            + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))
            * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

        // Compute the denominator of the Equation (2)
        const float l12 = std::sqrt(
            (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            + (z1 - z2) * (z1 - z2));

        // Compute the normal vector (la, lb, lc) from the projection of
        // the point `i` on the edge line between points `j` and `l` and
        // the point `i`
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

        // Compute the d_e using the Equation (2), which is the point-to-edge
        // distance between the point `i` and the edge line between points
        // `j` and `l` in the previous scan
        const float ld2 = a012 / l12;

        // Compute the projection of the point `i` on the edge line
        // between points `j` and `l` using the above normal vector
        pcl::PointXYZI pointProj = pointSel;
        pointProj.x -= la * ld2;
        pointProj.y -= lb * ld2;
        pointProj.z -= lc * ld2;

        // Assign smaller weights for the points with larger
        // point-to-edge distances and zero weights for outliers
        // with distances larger than the threshold (Section V.D)
        const float s = iterCount < 5 ? 1.0f : (1.0f - 1.8f * std::fabs(ld2));

        pcl::PointXYZI coeff;
        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
        coeff.intensity = s * ld2;

        // Store the coefficient vector and the original point `i`
        // that is not reprojected to the beginning of the current sweep
        if (s <= 0.1f || ld2 == 0.0f)
            continue;

        this->_laserCloudOri->push_back(this->_cornerPointsSharp->points[i]);
        this->_coeffSel->push_back(coeff);
    }
}

// Find point-to-edge correspondences from the corner point cloud
void BasicLaserOdometry::findCornerCorrespondence()
{
    const std::size_t cornerPointsSharpNum =
        this->_cornerPointsSharp->points.size();
    this->_pointSearchCornerInd1.resize(cornerPointsSharpNum);
    this->_pointSearchCornerInd2.resize(cornerPointsSharpNum);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(
        *this->_lastCornerCloud, *this->_lastCornerCloud, indices);

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
        pcl::PointXYZI pointSel;
        this->transformToStart(this->_cornerPointsSharp->points[i], pointSel);

        // Find the closest point in the last scan for `pointSel`,
        // which is the point `j` in the paper
        this->_lastCornerKDTree.nearestKSearch(
            pointSel, 1, pointSearchInd, pointSearchSqDis);

        // If the distance between the corner point in the current scan
        // (point `i` in the paper) and its closest point in the last
        // scan (point `j` in the paper) is larger than 5 meters, then the
        // correspondence for the current corner point `i` is not found
        if (pointSearchSqDis[0] >= 25.0f) {
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
        float minPointSqDis2 = 25.0f;

        // The below should be `j < lastCornerCloudSize` instead of
        // `j < cornerPointsSharpNum`
        for (int j = closestPointInd + 1; j < cornerPointsSharpNum; ++j) {
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
    }
}

// Compute the distances and coefficients from the point-to-plane
// correspondences
void BasicLaserOdometry::computePlaneDistances(int iterCount)
{
    if (iterCount % 5 == 0)
        this->findPlaneCorrespondence();

    const std::size_t surfPointsFlatNum =
        this->_surfPointsFlat->points.size();

    // For each planar point in the current scan (stored in `_surfPointsFlat`),
    // find the closest neighbor point in the last scan (stored in
    // `_lastSurfaceCloud`) which is reprojected to the beginning of the
    // current sweep (i.e., timestamp of the current scan)
    for (int i = 0; i < surfPointsFlatNum; ++i) {
        // Reproject the planar point in the current scan to the beginning
        // of the current sweep (point `i` in the paper)
        pcl::PointXYZI pointSel;
        this->transformToStart(this->_surfPointsFlat->points[i], pointSel);

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

        // Compute the vector (pa, pb, pc) that is perpendicular to the
        // plane defined by points `j`, `l`, and `m`, which is written as
        // (X^L_(k, j) - X^L_(k, l)) * (X^L_(k, j) - X^L_(k, m))
        float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z)
                   - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
        float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x)
                   - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
        float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y)
                   - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
        // Compute the inner product using X^L_(k, j) and the above
        // vector (X^L_(k, j) - X^L_(k, l)) * (X^L_(k, j) - X^L_(k, m))
        float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

        // Compute the denominator of the Equation (3)
        const float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // Compute the d_h using the Equation (3), which is the point-to-plane
        // distance between the point `i` and the plane defined by three points
        // `j`, `l`, and `m`
        // Note that the distance below could be negative
        const float pd2 = pa * pointSel.x + pb * pointSel.y
                          + pc * pointSel.z + pd;

        // Compute the projection of the point `i` on the plane defined by
        // points `j`, `l`, and `m` using the normal vector
        // If the normal vector (pa, pb, pc) and the vector between `i` and `j`
        // point to the same direction, `pd2` is positive and `pointSel` is
        // moved to the opposite direction of the normal vector; otherwise,
        // two vectors point to the opposite direction and `pd2` is negative,
        // which means that `pointSel` is moved to the direction of the normal
        // vector and produces the correct result
        pcl::PointXYZI pointProj;
        pointProj = pointSel;
        pointProj.x -= pa * pd2;
        pointProj.y -= pb * pd2;
        pointProj.z -= pc * pd2;

        // Assign smaller weights for the points with larger
        // point-to-plane distances and zero weights for outliers
        // with distances larger than the threshold (Section V.D)
        const float s = iterCount < 5 ? 1.0f :
            (1.0f - 1.8f * std::fabs(pd2)
            / std::sqrt(calcPointDistance(pointSel)));

        pcl::PointXYZI coeff;
        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        // Store the coefficient vector and the original point `i`
        // that is not reprojected to the beginning of the current sweep
        if (s <= 0.1f || pd2 == 0.0f)
            continue;

        this->_laserCloudOri->push_back(this->_surfPointsFlat->points[i]);
        this->_coeffSel->push_back(coeff);
    }
}

// Find point-to-plane correspondences from the planar point cloud
void BasicLaserOdometry::findPlaneCorrespondence()
{
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
        pcl::PointXYZI pointSel;
        this->transformToStart(this->_surfPointsFlat->points[i], pointSel);

        // Find the closest point in the last scan for `pointSel`,
        // which is the point `j` in the paper
        this->_lastSurfaceKDTree.nearestKSearch(
            pointSel, 1, pointSearchInd, pointSearchSqDis);

        // If the distance between the planar point in the current scan
        // (point `i` in the paper) and its closest point in the last
        // scan (point `j` in the paper) is larger than 5 meters, then the
        // correspondence for the current planar point `i` is not found
        if (pointSearchSqDis[0] >= 25.0f) {
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
        float minPointSqDis2 = 25.0f;
        float minPointSqDis3 = 25.0f;

        // The below should be `j < _lastSurfaceCloud` instead of
        // `j < surfPointsFlatNum`
        for (int j = closestPointInd + 1; j < surfPointsFlatNum; ++j) {
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
    }
}

} // namespace loam
