
// BasicScanRegistration.cpp

#include <algorithm>
#include <pcl/filters/voxel_grid.h>

#include "loam_velodyne/BasicScanRegistration.h"
#include "loam_velodyne/MathUtils.h"

namespace loam {

void BasicScanRegistration::processScanlines(
    const Time& scanTime,
    const std::vector<pcl::PointCloud<pcl::PointXYZI>>& laserCloudScans)
{
    // Reset internal buffers and set IMU start state based on current scan time
    this->reset(scanTime);

    // Construct sorted full resolution cloud
    std::size_t cloudSize = 0;

    for (std::size_t i = 0; i < laserCloudScans.size(); ++i) {
        this->_laserCloud += laserCloudScans[i];

        IndexRange range { cloudSize, 0 };
        cloudSize += laserCloudScans[i].size();
        range.second = cloudSize > 0 ? cloudSize - 1 : 0;
        this->_scanIndices.push_back(range);
    }

    // Collect the metrics
    this->_metricsMsg.sweep_start_stamp = toROSTime(this->_sweepStart);
    this->_metricsMsg.num_of_full_res_points = this->_laserCloud.size();

    this->_metricsMsg.num_of_points_in_rings.clear();
    std::transform(laserCloudScans.begin(), laserCloudScans.end(),
                   std::back_inserter(this->_metricsMsg.num_of_points_in_rings),
                   [](const pcl::PointCloud<pcl::PointXYZI>& cloud) {
                       return static_cast<std::uint32_t>(cloud.size()); });

    this->extractFeatures();
    this->updateIMUTransform();
}

bool BasicScanRegistration::configure(const RegistrationParams& config)
{
    this->_config = config;
    this->_imuHistory.ensureCapacity(this->_config.imuHistorySize);
    return true;
}

void BasicScanRegistration::reset(const Time& scanTime)
{
    this->_scanTime = scanTime;

    // Re-initialize IMU start index and state
    this->_imuIdx = 0;

    if (this->hasIMUData())
        this->interpolateIMUStateFor(0.0f, this->_imuStart);

    // Clear internal cloud buffers at the beginning of a sweep
    this->_sweepStart = scanTime;

    // Clear cloud buffers
    this->_laserCloud.clear();
    this->_cornerPointsSharp.clear();
    this->_cornerPointsLessSharp.clear();
    this->_surfacePointsFlat.clear();
    this->_surfacePointsLessFlat.clear();

    // Clear scan indices vector
    this->_scanIndices.clear();
}

void BasicScanRegistration::updateIMUData(Vector3& acc, IMUState& newState)
{
    // TODO: Argument `acc` is not needed since `newState` already has
    // the acceleration from the IMU

    if (!this->_imuHistory.empty()) {
        // Accumulate IMU position and velocity over time
        // `newState` represents the rotation in fixed XYZ Euler angles
        rotateZXY(acc, newState.roll, newState.pitch, newState.yaw);

        // Compute IMU position and velocity using integration
        const IMUState& prevState = this->_imuHistory.last();
        const float timeDiff = toSec(newState.stamp - prevState.stamp);
        newState.position = prevState.position
                            + prevState.velocity * timeDiff
                            + 0.5 * acc * timeDiff * timeDiff;
        newState.velocity = prevState.velocity + acc * timeDiff;
    }

    this->_imuHistory.push(newState);
}

void BasicScanRegistration::projectPointToStartOfSweep(
    pcl::PointXYZI& point, float relTime)
{
    // Project point to the start of the sweep using corresponding IMU data
    if (this->hasIMUData()) {
        // Since this method is called from `MultiScanRegistration::process()`,
        // `_scanTime` should be updated with the latest scan timestamp
        this->setIMUTransformFor(relTime);
        this->transformToStartIMU(point);
    }
}

void BasicScanRegistration::setIMUTransformFor(const float relTime)
{
    // When this method is called from `MultiScanRegistration::process()`,
    // `_scanTime` is the timestamp of the previous scan, which may be wrong and
    // timestamp of the latest scan should be used instead
    // Before this method is called from `MultiScanRegistration::process()`,
    // `_scanTime` should be updated with the latest scan timestamp
    this->interpolateIMUStateFor(relTime, this->_imuCur);

    // Both `_scanTime` and `_sweepStart` represent the timestamp of the
    // current scan and `_scanTime - _sweepStart` is always zero
    // `relTime` may be used instead of `relSweepTime`
    float relSweepTime = toSec(this->_scanTime - this->_sweepStart) + relTime;
    // `interpolatedState` is the expected IMU state if the LiDAR moved at
    // constant speed (linear motion) during the current scan
    const auto interpolatedState = this->_imuStart.position
                                   + this->_imuStart.velocity * relSweepTime;
    // `_imuPositionShift` is the difference between the accumulated IMU
    // position and the interpolated IMU position which represents the
    // position shift caused by acceleration or deceleration (nonlinear motion)
    this->_imuPositionShift = this->_imuCur.position - interpolatedState;
}

void BasicScanRegistration::transformToStartIMU(pcl::PointXYZI& point)
{
    // Rotate point to the global coordinate frame
    // `_imuCur` represents the rotation in fixed XYZ Euler angles
    rotateZXY(point, this->_imuCur.roll,
              this->_imuCur.pitch, this->_imuCur.yaw);

    // Add the position shift in global coordinate frame
    point.x += this->_imuPositionShift.x();
    point.y += this->_imuPositionShift.y();
    point.z += this->_imuPositionShift.z();

    // Rotate point back to the local IMU coordinate frame
    // relative to the start IMU state
    rotateYXZ(point, -this->_imuStart.yaw,
              -this->_imuStart.pitch, -this->_imuStart.roll);
}

void BasicScanRegistration::interpolateIMUStateFor(
    const float &relTime, IMUState& outputState)
{
    auto imuTime = [this](const std::size_t idx) {
        return this->_imuHistory[this->_imuIdx].stamp; };

    // `timeDiff` should be negative to perform the interpolation, as it
    // represents the time difference from the time of the current scan point
    // specified by `relTime` to the timestamp of the latest IMU data
    double timeDiff = toSec(this->_scanTime - imuTime(this->_imuIdx)) + relTime;

    while (this->_imuIdx < this->_imuHistory.size() - 1 && timeDiff > 0)
        timeDiff = toSec(this->_scanTime - imuTime(++this->_imuIdx)) + relTime;

    if (this->_imuIdx == 0 || timeDiff > 0) {
        // Use the latest IMU data since IMU data newer than the acquisition
        // of the current scan point is not available
        outputState = this->_imuHistory[this->_imuIdx];
    } else {
        const double imuTimeDiff = toSec(imuTime(this->_imuIdx)
                                         - imuTime(this->_imuIdx - 1));
        const float ratio = -timeDiff / imuTimeDiff;
        IMUState::interpolate(this->_imuHistory[this->_imuIdx],
                              this->_imuHistory[this->_imuIdx - 1],
                              ratio, outputState);
    }
}

void BasicScanRegistration::extractFeatures(const std::uint16_t beginIdx)
{
    const ros::Time startTime = ros::Time::now();

    // Extract features from individual scans
    const std::size_t nScans = this->_scanIndices.size();

    std::size_t numOfOriginalLessFlatPoints = 0;

    for (std::size_t i = beginIdx; i < nScans; ++i) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlatScan {
            new pcl::PointCloud<pcl::PointXYZI>() };
        const std::size_t scanStartIdx = this->_scanIndices[i].first;
        const std::size_t scanEndIdx = this->_scanIndices[i].second;

        // Skip empty scans
        if (scanEndIdx <= scanStartIdx + 2 * this->_config.curvatureRegion)
            continue;

        const std::size_t scanSize =
            scanEndIdx - scanStartIdx - 2 * this->_config.curvatureRegion;

        // Reset scan buffers
        this->setScanBuffersFor(scanStartIdx, scanEndIdx);

        // Extract features from equally sized scan regions
        for (int j = 0; j < this->_config.nFeatureRegions; ++j) {
            const std::size_t regionStartIdx =
                (scanStartIdx + this->_config.curvatureRegion)
                + scanSize * j / this->_config.nFeatureRegions;
            const std::size_t regionEndIdx =
                (scanStartIdx + this->_config.curvatureRegion)
                + scanSize * (j + 1) / this->_config.nFeatureRegions - 1;

            // Skip empty regions
            if (regionEndIdx <= regionStartIdx)
                continue;

            const std::size_t regionSize = regionEndIdx - regionStartIdx + 1;

            // Reset region buffers
            this->setRegionBuffersFor(regionStartIdx, regionEndIdx);

            // Extract corner features
            for (int k = regionSize - 1, largestPickedNum = 0; k >= 0 &&
                 largestPickedNum < this->_config.maxCornerLessSharp; --k) {
                const std::size_t idx = this->_regionSortIndices[k];
                const std::size_t scanIdx = idx - scanStartIdx;
                const std::size_t regionIdx = idx - regionStartIdx;

                if (this->_scanNeighborPicked[scanIdx] != 0 ||
                    this->_regionCurvature[regionIdx]
                    <= this->_config.surfaceCurvatureThreshold)
                    continue;

                largestPickedNum++;

                if (largestPickedNum <= this->_config.maxCornerSharp) {
                    this->_regionLabel[regionIdx] = CORNER_SHARP;
                    this->_cornerPointsSharp.push_back(this->_laserCloud[idx]);
                } else {
                    this->_regionLabel[regionIdx] = CORNER_LESS_SHARP;
                }

                this->_cornerPointsLessSharp.push_back(this->_laserCloud[idx]);

                this->markAsPicked(idx, scanIdx);
            }

            // Extract flat surface features
            for (int k = 0, smallestPickedNum = 0; k < regionSize &&
                 smallestPickedNum < this->_config.maxSurfaceFlat; ++k) {
                const std::size_t idx = this->_regionSortIndices[k];
                const std::size_t scanIdx = idx - scanStartIdx;
                const std::size_t regionIdx = idx - regionStartIdx;

                if (this->_scanNeighborPicked[scanIdx] != 0 ||
                    this->_regionCurvature[regionIdx]
                    >= this->_config.surfaceCurvatureThreshold)
                    continue;

                smallestPickedNum++;
                this->_regionLabel[regionIdx] = SURFACE_FLAT;
                this->_surfacePointsFlat.push_back(this->_laserCloud[idx]);

                this->markAsPicked(idx, scanIdx);
            }

            // Extract less flat surface features
            for (int k = 0; k < regionSize; ++k) {
                if (this->_regionLabel[k] <= SURFACE_LESS_FLAT)
                    surfPointsLessFlatScan->push_back(
                        this->_laserCloud[regionStartIdx + k]);
            }
        }

        // Downsample less flat surface point cloud of current scan
        pcl::PointCloud<pcl::PointXYZI> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(this->_config.lessFlatFilterSize,
                                   this->_config.lessFlatFilterSize,
                                   this->_config.lessFlatFilterSize);
        downSizeFilter.filter(surfPointsLessFlatScanDS);
        this->_surfacePointsLessFlat += surfPointsLessFlatScanDS;
        numOfOriginalLessFlatPoints += surfPointsLessFlatScan->size();
    }

    // Collect the metrics
    const ros::Time endTime = ros::Time::now();
    this->_metricsMsg.feature_extraction_time = endTime - startTime;
    this->_metricsMsg.num_of_sharp_points =
        this->_cornerPointsSharp.size();
    this->_metricsMsg.num_of_less_sharp_points =
        this->_cornerPointsLessSharp.size();
    this->_metricsMsg.num_of_flat_points =
        this->_surfacePointsFlat.size();
    this->_metricsMsg.num_of_less_flat_points_before_ds =
        numOfOriginalLessFlatPoints;
    this->_metricsMsg.num_of_less_flat_points =
        this->_surfacePointsLessFlat.size();
}

void BasicScanRegistration::updateIMUTransform()
{
    // `_imuStart` stores the IMU state when the first point in `_laserCloud` is
    // obtained (i.e., IMU state at the scan start time)
    this->_imuTrans[0].x = this->_imuStart.pitch.rad();
    this->_imuTrans[0].y = this->_imuStart.yaw.rad();
    this->_imuTrans[0].z = this->_imuStart.roll.rad();

    // `_imuCur` is only updated when `projectPointToStartOfSweep()` is called
    // from `MultiScanRegistration::process()`, meaning that _imuCur stores
    // the IMU state when the last point in `_laserCloud` is obtained
    // (i.e., IMU state at the scan end time)
    this->_imuTrans[1].x = this->_imuCur.pitch.rad();
    this->_imuTrans[1].y = this->_imuCur.yaw.rad();
    this->_imuTrans[1].z = this->_imuCur.roll.rad();

    // `_imuPositionShift` is the position shift from the interpolated IMU
    // position to the accumulated IMU position
    Vector3 imuShiftFromStart = this->_imuPositionShift;
    // Rotate to the local coordinate frame of the IMU start state
    rotateYXZ(imuShiftFromStart, -this->_imuStart.yaw,
              -this->_imuStart.pitch, -this->_imuStart.roll);

    this->_imuTrans[2].x = imuShiftFromStart.x();
    this->_imuTrans[2].y = imuShiftFromStart.y();
    this->_imuTrans[2].z = imuShiftFromStart.z();

    // `imuVelocityFromStart` is the IMU velocity difference between the
    // scan start time and scan end time
    Vector3 imuVelocityFromStart = this->_imuCur.velocity
                                   - this->_imuStart.velocity;
    // Rotate to the local coordinate frame of the IMU start state
    rotateYXZ(imuVelocityFromStart, -this->_imuStart.yaw,
              -this->_imuStart.pitch, -this->_imuStart.roll);

    this->_imuTrans[3].x = imuVelocityFromStart.x();
    this->_imuTrans[3].y = imuVelocityFromStart.y();
    this->_imuTrans[3].z = imuVelocityFromStart.z();
}

void BasicScanRegistration::setRegionBuffersFor(
    const std::size_t& startIdx, const std::size_t& endIdx)
{
    // Resize buffers
    const std::size_t regionSize = endIdx - startIdx + 1;
    this->_regionCurvature.resize(regionSize);
    this->_regionSortIndices.resize(regionSize);
    this->_regionLabel.assign(regionSize, SURFACE_LESS_FLAT);

    // Compute point curvatures and reset sort indices
    const float pointWeight = -2.0f * this->_config.curvatureRegion;

    for (std::size_t i = startIdx, regionIdx = 0;
         i <= endIdx; ++i, ++regionIdx) {
        float diffX = pointWeight * this->_laserCloud[i].x;
        float diffY = pointWeight * this->_laserCloud[i].y;
        float diffZ = pointWeight * this->_laserCloud[i].z;

        for (int j = 1; j <= this->_config.curvatureRegion; ++j) {
            diffX += this->_laserCloud[i + j].x + this->_laserCloud[i - j].x;
            diffY += this->_laserCloud[i + j].y + this->_laserCloud[i - j].y;
            diffZ += this->_laserCloud[i + j].z + this->_laserCloud[i - j].z;
        }

        this->_regionCurvature[regionIdx] =
            diffX * diffX + diffY * diffY + diffZ * diffZ;
        this->_regionSortIndices[regionIdx] = i;
    }

    // Sort point curvatures
    std::sort(this->_regionSortIndices.begin(), this->_regionSortIndices.end(),
              [&](const std::size_t lhs, const std::size_t rhs) {
                  return this->_regionCurvature[lhs - startIdx]
                         > this->_regionCurvature[rhs - startIdx]; });

    /* for (size_t i = 1; i < regionSize; ++i) {
        for (size_t j = i; j >= 1; --j) {
            const std::size_t idx0 = this->_regionSortIndices[j] - startIdx;
            const std::size_t idx1 = this->_regionSortIndices[j - 1] - startIdx;
            if (this->_regionCurvature[idx0] < this->_regionCurvature[idx1]) {
                std::swap(this->_regionSortIndices[j],
                          this->_regionSortIndices[j - 1]);
            }
        }
    } */
}

void BasicScanRegistration::setScanBuffersFor(
    const std::size_t& startIdx, const std::size_t& endIdx)
{
    // Resize buffers
    const std::size_t scanSize = endIdx - startIdx + 1;
    this->_scanNeighborPicked.assign(scanSize, 0);

    // Mark unreliable points as picked
    for (std::size_t i = startIdx + this->_config.curvatureRegion;
         i < endIdx - this->_config.curvatureRegion; ++i) {
        const pcl::PointXYZI& previousPoint = this->_laserCloud[i - 1];
        const pcl::PointXYZI& point = this->_laserCloud[i];
        const pcl::PointXYZI& nextPoint = this->_laserCloud[i + 1];

        const float diffNext = calcSquaredDiff(nextPoint, point);

        if (diffNext > 0.1) {
            const float depth1 = calcPointDistance(point);
            const float depth2 = calcPointDistance(nextPoint);

            if (depth1 > depth2) {
                const float weightedDistance = std::sqrt(calcSquaredDiff(
                    nextPoint, point, depth2 / depth1)) / depth2;

                if (weightedDistance < 0.1) {
                    const std::size_t unreliableIdx =
                        i - startIdx - this->_config.curvatureRegion;
                    std::fill_n(&this->_scanNeighborPicked[unreliableIdx],
                                this->_config.curvatureRegion + 1, 1);
                    continue;
                }
            } else {
                const float weightedDistance = std::sqrt(calcSquaredDiff(
                    point, nextPoint, depth1 / depth2)) / depth1;

                if (weightedDistance < 0.1)
                    std::fill_n(&this->_scanNeighborPicked[i - startIdx + 1],
                                this->_config.curvatureRegion + 1, 1);
            }
        }

        const float diffPrevious = calcSquaredDiff(point, previousPoint);
        const float distSq = calcSquaredPointDistance(point);

        if (diffNext > 0.0002 * distSq && diffPrevious > 0.0002 * distSq)
            this->_scanNeighborPicked[i - startIdx] = 1;
    }
}

void BasicScanRegistration::markAsPicked(
    const std::size_t& cloudIdx, const std::size_t& scanIdx)
{
    this->_scanNeighborPicked[scanIdx] = 1;

    for (int i = 1; i <= this->_config.curvatureRegion; ++i) {
        // `point` could be `_laserCloud[cloudIdx]`
        const auto& point = this->_laserCloud[cloudIdx + i];
        const auto& prevPoint = this->_laserCloud[cloudIdx + i - 1];

        if (calcSquaredDiff(point, prevPoint) > 0.05)
            break;

        this->_scanNeighborPicked[scanIdx + i] = 1;
    }

    for (int i = 1; i <= this->_config.curvatureRegion; ++i) {
        // `point` could be `_laserCloud[cloudIdx]`
        const auto& point = this->_laserCloud[cloudIdx - i];
        const auto& nextPoint = this->_laserCloud[cloudIdx - i + 1];

        if (calcSquaredDiff(point, nextPoint) > 0.05)
            break;

        this->_scanNeighborPicked[scanIdx - i] = 1;
    }
}

// Clear the metrics message
void BasicScanRegistration::clearMetricsMsg()
{
    this->_metricsMsg.point_extraction_time = ros::Duration(0.0);
    this->_metricsMsg.point_cloud_stamp = ros::Time(0.0);
    this->_metricsMsg.num_of_unprocessed_points = 0;

    this->_metricsMsg.sweep_start_stamp = ros::Time(0.0);
    this->_metricsMsg.num_of_full_res_points = 0;
    this->_metricsMsg.num_of_points_in_rings.clear();

    this->_metricsMsg.feature_extraction_time = ros::Duration(0.0);
    this->_metricsMsg.num_of_sharp_points = 0;
    this->_metricsMsg.num_of_less_sharp_points = 0;
    this->_metricsMsg.num_of_flat_points = 0;
    this->_metricsMsg.num_of_less_flat_points_before_ds = 0;
    this->_metricsMsg.num_of_less_flat_points = 0;
}

} // namespace loam
