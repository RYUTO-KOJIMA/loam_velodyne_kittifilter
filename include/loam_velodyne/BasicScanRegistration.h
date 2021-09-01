
// BasicScanRegistration.h

#pragma once

#include <utility>
#include <vector>

#include <pcl/point_cloud.h>

#include "common.h"
#include "Angle.h"
#include "Vector3.h"
#include "CircularBuffer.h"

namespace loam {

/** \brief A pair describing the start end end index of a range. */
using IndexRange = std::pair<std::size_t, std::size_t>;

/* Point label options. */
enum PointLabel
{
    // Sharp corner point
    CORNER_SHARP = 2,
    // Less sharp corner point
    CORNER_LESS_SHARP = 1,
    // Less flat surface point
    SURFACE_LESS_FLAT = 0,
    // Flat surface point
    SURFACE_FLAT = -1
};

/* Scan Registration configuration parameters. */
struct RegistrationParams
{
    RegistrationParams(const float scanPeriod_ = 0.1,
                       const int imuHistorySize_ = 200,
                       const int nFeatureRegions_ = 6,
                       const int curvatureRegion_ = 5,
                       const int maxCornerSharp_ = 2,
                       const int maxSurfaceFlat_ = 4,
                       const float lessFlatFilterSize_ = 0.2,
                       const float surfaceCurvatureThreshold_ = 0.1) :
        scanPeriod(scanPeriod_),
        imuHistorySize(imuHistorySize_),
        nFeatureRegions(nFeatureRegions_),
        curvatureRegion(curvatureRegion_),
        maxCornerSharp(maxCornerSharp_),
        maxCornerLessSharp(10 * maxCornerSharp_),
        maxSurfaceFlat(maxSurfaceFlat_),
        lessFlatFilterSize(lessFlatFilterSize_),
        surfaceCurvatureThreshold(surfaceCurvatureThreshold_) { }

    /* The time per scan. */
    float scanPeriod;
    /* The size of the IMU history state buffer. */
    int   imuHistorySize;
    /* The number of (equally sized) regions used to distribute
     * the feature extraction within a scan. */
    int   nFeatureRegions;
    /* The number of surrounding points (+/- region around a point)
     * used to calculate a point curvature. */
    int   curvatureRegion;
    /* The maximum number of sharp corner points per feature region. */
    int   maxCornerSharp;
    /* The maximum number of less sharp corner points per feature region. */
    int   maxCornerLessSharp;
    /* The maximum number of flat surface points per feature region. */
    int   maxSurfaceFlat;
    /* The voxel size used for down sizing the remaining
     * less flat surface points. */
    float lessFlatFilterSize;
    /* The curvature threshold below / above a point is considered
     * a flat / corner point. */
    float surfaceCurvatureThreshold;
};

/* IMU state data. */
struct IMUState
{
    /* The time of the measurement leading to this state (in seconds). */
    Time    stamp;
    /* The current roll angle. */
    Angle   roll;
    /* The current pitch angle. */
    Angle   pitch;
    /* The current yaw angle. */
    Angle   yaw;
    /* The accumulated global IMU position in 3D space. */
    Vector3 position;
    /* The accumulated global IMU velocity in 3D space. */
    Vector3 velocity;
    /* The current (local) IMU acceleration in 3D space. */
    Vector3 acceleration;

    /** \brief Interpolate between two IMU states.
     *
     * @param start The first IMUState
     * @param end The second IMUState
     * @param ratio The interpolation ratio
     * @param result The target for storing the interpolation result
     */
    static void interpolate(const IMUState& start, const IMUState& end,
                            const float& ratio, IMUState& result)
    {
        const float invRatio = 1.0f - ratio;

        result.roll = start.roll.rad() * invRatio + end.roll.rad() * ratio;
        result.pitch = start.pitch.rad() * invRatio + end.pitch.rad() * ratio;

        if (start.yaw.rad() - end.yaw.rad() > M_PI)
            result.yaw = start.yaw.rad() * invRatio
                         + (end.yaw.rad() + 2.0f * M_PI) * ratio;
        else if (start.yaw.rad() - end.yaw.rad() < -M_PI)
            result.yaw = start.yaw.rad() * invRatio
                         + (end.yaw.rad() - 2.0f * M_PI) * ratio;
        else
            result.yaw = start.yaw.rad() * invRatio + end.yaw.rad() * ratio;

        result.velocity = start.velocity * invRatio + end.velocity * ratio;
        result.position = start.position * invRatio + end.position * ratio;
    }
};

class BasicScanRegistration
{
public:
    /** \brief Process a new cloud as a set of scanlines.
     *
     * @param relTime The time relative to the scan time
     */
    void processScanlines(
        const Time& scanTime,
        const std::vector<pcl::PointCloud<pcl::PointXYZI>>& laserCloudScans);

    bool configure(const RegistrationParams& config = RegistrationParams()); 

    /** \brief Update new IMU state (also mutates the arguments). */
    void updateIMUData(Vector3& acc, IMUState& newState);

    /** \brief Project a point to the start of the sweep
     * using corresponding IMU data.
     *
     * @param point The point to modify
     * @param relTime The time to project by
     */
    void projectPointToStartOfSweep(pcl::PointXYZI& point, float relTime);

    inline const auto& config() const { return this->_config; }
    inline const auto& imuTransform() const { return this->_imuTrans; }
    inline const auto& sweepStart() const { return this->_sweepStart; }

    inline const auto& laserCloud() const
    { return this->_laserCloud; }
    inline const auto& cornerPointsSharp() const
    { return this->_cornerPointsSharp; }
    inline const auto& cornerPointsLessSharp() const
    { return this->_cornerPointsLessSharp; }
    inline const auto& surfacePointsFlat() const
    { return this->_surfacePointsFlat; }
    inline const auto& surfacePointsLessFlat() const
    { return this->_surfacePointsLessFlat; }

private:
    /** \brief Check is IMU data is available. */
    inline bool hasIMUData() { return !this->_imuHistory.empty(); }

    /** \brief Set up the current IMU transformation
     * for the specified relative time.
     *
     * @param relTime The time relative to the scan time
     */
    void setIMUTransformFor(const float relTime);

    /** \brief Project the given point to the start of the sweep,
     * using the current IMU state and position shift.
     *
     * @param point The point to project
     */
    void transformToStartIMU(pcl::PointXYZI& point);

    /** \brief Prepare for next scan / sweep.
     *
     * @param scanTime The current scan time
     * @param newSweep Indicator if a new sweep has started
     */
    void reset(const Time& scanTime);

    /** \brief Extract features from current laser cloud.
     *
     * @param beginIdx The index of the first scan to extract features from
     */
    void extractFeatures(const std::uint16_t beginIdx = 0);

    /** \brief Set up region buffers for the specified point range.
     *
     * @param startIdx The region start index
     * @param endIdx The region end index
     */
    void setRegionBuffersFor(const std::size_t& startIdx,
                             const std::size_t& endIdx);

    /** \brief Set up scan buffers for the specified point range.
     *
     * @param startIdx The scan start index
     * @param endIdx The scan end index
     */
    void setScanBuffersFor(const std::size_t& startIdx,
                           const std::size_t& endIdx);

    /** \brief Mark a point and its neighbors as picked.
     *
     * This method will mark neighboring points within the curvature region
     * as picked, as long as they remain within close distance to each other.
     *
     * @param cloudIdx The index of the picked point in the
     * full resolution cloud
     * @param scanIdx The index of the picked point relative to the current scan
     */
    void markAsPicked(const std::size_t& cloudIdx,
                      const std::size_t& scanIdx);

    /** \brief Try to interpolate the IMU state for the given time.
     *
     * @param relTime The time relative to the scan time
     * @param outputState The output state instance
     */
    void interpolateIMUStateFor(const float& relTime, IMUState& outputState);

    void updateIMUTransform();

private:
    // Registration parameters
    RegistrationParams _config;

    // Full resolution input cloud
    pcl::PointCloud<pcl::PointXYZI> _laserCloud;
    // Start and end indices of the individual scans
    // within the full resolution cloud
    std::vector<IndexRange> _scanIndices;

    // Sharp corner points cloud
    pcl::PointCloud<pcl::PointXYZI> _cornerPointsSharp;
    // Less sharp corner points cloud
    pcl::PointCloud<pcl::PointXYZI> _cornerPointsLessSharp;
    // Flat surface points cloud
    pcl::PointCloud<pcl::PointXYZI> _surfacePointsFlat;
    // Less flat surface points cloud
    pcl::PointCloud<pcl::PointXYZI> _surfacePointsLessFlat;

    // Time stamp of beginning of current sweep
    Time _sweepStart;
    // Time stamp of most recent scan (same as the beginning of current sweep)
    Time _scanTime;
    // The interpolated IMU state corresponding to the start time of
    // the currently processed laser scan
    IMUState _imuStart;
    // The interpolated IMU state corresponding to the time of
    // the currently processed laser scan point
    IMUState _imuCur;
    // Position shift between accumulated IMU position and interpolated IMU
    // position caused by acceleration or deceleration, i.e., nonlinear motion
    Vector3 _imuPositionShift;
    // The current index in the IMU history
    std::size_t _imuIdx = 0;
    // History of IMU states for cloud registration
    CircularBuffer<IMUState> _imuHistory;

    // IMU transformation information (contains 4 transformations)
    pcl::PointCloud<pcl::PointXYZ> _imuTrans = { 4, 1 };

    // Point curvature buffer
    std::vector<float> _regionCurvature;
    // Point label buffer
    std::vector<PointLabel> _regionLabel;
    // Sorted region indices based on point curvature
    std::vector<std::size_t> _regionSortIndices;
    // Flag if neighboring point was already picked
    std::vector<int> _scanNeighborPicked;
};

} // namespace loam
