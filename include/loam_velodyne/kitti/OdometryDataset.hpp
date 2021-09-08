
// KittiOdometryDataset.hpp

#ifndef LOAM_KITTI_ODOMETRY_DATASET_HPP
#define LOAM_KITTI_ODOMETRY_DATASET_HPP

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ROS Melodic supports C++14
#include <experimental/filesystem>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ros/ros.h>

namespace loam {
namespace kitti {

namespace fs = std::experimental::filesystem;

// Type definition for the calibration data
using CalibrationData = std::unordered_map<std::string, std::vector<double>>;

class KittiOdometryDataset final
{
public:
    // Constructor
    KittiOdometryDataset() = default;
    // Destructor
    ~KittiOdometryDataset() = default;

    // Load the odometry dataset
    bool Load(const std::string& path, const int sequence);

    // Load the velodyne point cloud at the specified index
    bool PointCloudAt(const std::size_t idx,
                      pcl::PointCloud<pcl::PointXYZI>& pointCloud);

    // Get the path to the Kitti odometry dataset
    inline const fs::path& DatasetPath() const { return this->mDatasetPath; }
    // Get the path to the sequence
    inline const fs::path& SequencePath() const { return this->mSequencePath; }
    // Get the path to the ground-truth poses
    inline const fs::path& PosePath() const { return this->mPosePath; }
    // Get the sequence string (00 to 21)
    inline const std::string& SequenceStr() const { return this->mSequenceStr; }
    // Get the sequence number (0 to 21)
    inline int Sequence() const { return this->mSequence; }

    // Get the number of the data
    inline std::size_t NumOfData() const { return this->mTimestamps.size(); }

    // Get the transformation from the velodyne coordinate to the rectified
    // camera coordinate
    inline const Eigen::Matrix4d& TransformCamera0Velodyne() const
    { return this->mTransformCamera0Velodyne; }
    // Get the timestamps
    inline const std::vector<double>& Timestamps() const
    { return this->mTimestamps; }
    // Get the ground-truth poses
    inline const std::vector<Eigen::Matrix4d>& GroundTruthPoses() const
    { return this->mGroundTruthPoses; }

private:
    // Load the calibration data
    bool LoadCalibration();
    // Load the timestamps
    bool LoadTimestamps();
    // Load the ground-truth poses if available
    bool LoadGroundTruthPoses();
    // Load the velodyne point cloud file names
    bool LoadPointCloudFileNames();

private:
    // Path to the Kitti odometry dataset
    fs::path mDatasetPath;
    // Path to the sequence
    fs::path mSequencePath;
    // Path to the ground-truth poses
    fs::path mPosePath;
    // Sequence string (00 to 21)
    std::string mSequenceStr;
    // Sequence number (0 to 21)
    int mSequence;
    // Calibration data
    CalibrationData mCalibration;
    // Transformation from the velodyne to the rectified camera coordinate
    Eigen::Matrix4d mTransformCamera0Velodyne;
    // Timestamps
    std::vector<double> mTimestamps;
    // Ground-truth poses
    std::vector<Eigen::Matrix4d> mGroundTruthPoses;
    // Velodyne point cloud file names
    std::vector<fs::path> mPointCloudFileNames;
};

} // namespace kitti
} // namespace loam

#endif // LOAM_KITTI_ODOMETRY_DATASET_HPP
