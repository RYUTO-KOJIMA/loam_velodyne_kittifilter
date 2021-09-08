
// OdometryDataset.cpp

#include "loam_velodyne/kitti/OdometryDataset.hpp"

namespace loam {
namespace kitti {

// Load the odometry dataset
bool KittiOdometryDataset::Load(const std::string& path, const int sequence)
{
    // Check that the dataset directory exists
    if (!fs::is_directory(path)) {
        ROS_ERROR("Dataset path specified is invalid: %s", path.c_str());
        return false;
    }

    // Sequence number should be within 0 and 21
    if (sequence < 0 || sequence > 21) {
        ROS_ERROR("Dataset sequence should be within 0 and 21");
        return false;
    }

    // Convert the sequence number to the string
    std::ostringstream strStream;
    strStream << std::setw(2) << std::setfill('0') << sequence;
    const std::string sequenceStr = strStream.str();

    // Construct paths for the sequence and poses
    const fs::path datasetPath = path;
    const fs::path sequencePath = datasetPath / "sequences" / sequenceStr;
    const fs::path posePath = datasetPath / "poses";

    // Check that the sequence directory exists
    if (!fs::is_directory(sequencePath)) {
        ROS_ERROR("Directory for the sequence does not exist: %s",
                  sequencePath.c_str());
        return false;
    }

    // Check that the poses directory exists
    if (!fs::is_directory(posePath)) {
        ROS_ERROR("Directory for the poses does not exist: %s",
                  posePath.c_str());
        return false;
    }

    // Fill the members
    this->mDatasetPath = datasetPath;
    this->mSequencePath = sequencePath;
    this->mPosePath = posePath;
    this->mSequence = sequence;
    this->mSequenceStr = sequenceStr;

    // Load the calibration data
    if (!this->LoadCalibration())
        return false;

    // Load the timestamps
    if (!this->LoadTimestamps())
        return false;

    // Load the ground-truth poses if available
    if (!this->LoadGroundTruthPoses())
        return false;

    // Load the velodyne point cloud file names
    if (!this->LoadPointCloudFileNames())
        return false;

    // Check the number of timestamps and poses
    if (this->mGroundTruthPoses.size() > 0 &&
        this->mGroundTruthPoses.size() != this->mTimestamps.size()) {
        ROS_ERROR("Number of the ground-truth poses (%zu) is inconsistent "
                  "to the number of the timestamps (%zu)",
                  this->mGroundTruthPoses.size(), this->mTimestamps.size());
        return false;
    }

    // Check the number of timestamps and velodyne point clouds
    if (this->mPointCloudFileNames.size() != this->mTimestamps.size()) {
        ROS_ERROR("Number of the velodyne point clouds (%zu) is inconsistent "
                  "to the number of the timestamps (%zu)",
                  this->mPointCloudFileNames.size(), this->mTimestamps.size());
        return false;
    }

    return true;
}

// Load the point cloud from the file
bool KittiOdometryDataset::PointCloudAt(
    const std::size_t idx,
    pcl::PointCloud<pcl::PointXYZI>& pointCloud)
{
    // Check that the specified index is valid
    if (idx >= this->mPointCloudFileNames.size()) {
        ROS_ERROR("The specified index of the point cloud (%zu) "
                  "is out of range (%d to %zu)",
                  idx, 0, this->mPointCloudFileNames.size());
        return false;
    }

    // Get the file path to the specified point cloud
    const fs::path& fileName = this->mPointCloudFileNames[idx];

    // Check that the file path is consistent to the specified index
    std::ostringstream strStream;
    strStream << std::setw(6) << std::setfill('0') << idx << ".bin";
    const std::string expectedName = strStream.str();
    const std::string fileNameStr = fileName.string();

    if (fileNameStr.size() < expectedName.size() ||
        !std::equal(expectedName.rbegin(), expectedName.rend(),
                    fileNameStr.rbegin())) {
        ROS_ERROR("The point cloud file path is inconsistent to the "
                  "specified index (%zu): %s", idx, fileName.c_str());
        return false;
    }

    // Open the binary file
    std::ifstream pointCloudFile { fileName, std::ios::in | std::ios::binary };

    if (!pointCloudFile) {
        ROS_ERROR("Failed to open the point cloud file: %s", fileName.c_str());
        return false;
    }

    // Copy the file content to the buffer
    const std::vector<std::uint8_t> rawBuffer {
        std::istreambuf_iterator<char>(pointCloudFile),
        std::istreambuf_iterator<char>() };

    // Buffer size should be a multiply of sizeof(float) * 4
    if (rawBuffer.size() % (sizeof(float) * 4) != 0) {
        ROS_ERROR("The size of the point cloud (%zu) should be a multiply of "
                  "sizeof(float) * 4", rawBuffer.size());
        return false;
    }

    // Get the number of the points
    const std::size_t numOfPoints = rawBuffer.size() / (sizeof(float) * 4);

    // Load the point coordinates and reflectances
    pointCloud.clear();
    pointCloud.reserve(numOfPoints);

    const float* pBuffer = reinterpret_cast<const float*>(rawBuffer.data());
    const float* pX = pBuffer;
    const float* pY = pBuffer + 1;
    const float* pZ = pBuffer + 2;
    const float* pR = pBuffer + 3;

    for (std::size_t i = 0; i < numOfPoints; ++i) {
        // Append the new point
        pcl::PointXYZI point;
        point.x = *pX;
        point.y = *pY;
        point.z = *pZ;
        point.intensity = *pR;
        pointCloud.push_back(std::move(point));

        // Advance the pointer
        pX += 4;
        pY += 4;
        pZ += 4;
        pR += 4;
    }

    // Close the binary file
    pointCloudFile.close();

    return true;
}

// Load the calibration data from the file
bool KittiOdometryDataset::LoadCalibration()
{
    // Construct the file path to the calibration data
    const fs::path fileName = this->mSequencePath / "calib.txt";

    // Check that the file exists
    if (!fs::is_regular_file(fileName)) {
        ROS_ERROR("Calibration file does not exist: %s", fileName.c_str());
        return false;
    }

    // Open the calibration file
    std::ifstream calibrationFile { fileName };

    if (!calibrationFile) {
        ROS_ERROR("Failed to open the calibration file: %s", fileName.c_str());
        return false;
    }

    // Reset the calibration data
    this->mCalibration.clear();

    // Read the calibration file
    std::string calibrationStr;

    while (std::getline(calibrationFile, calibrationStr)) {
        // Split the entry with whitespaces
        std::istringstream strStream { calibrationStr };
        std::vector<std::string> tokens;
        std::copy(std::istream_iterator<std::string>(strStream),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(tokens));

        // Skip the empty line
        if (tokens.empty())
            continue;

        // Retrieve the entry Id (remove the trailing colon)
        const std::string& front = tokens.front();
        std::string entryId = front.substr(0, front.size() - 1);
        // Retrieve the calibration data
        std::vector<double> entryData;
        entryData.reserve(tokens.size() - 1);
        std::transform(std::next(tokens.begin()), tokens.end(),
                       std::back_inserter(entryData),
                       [](const std::string& str) { return std::stod(str); });

        // Insert the calibration data
        this->mCalibration.insert(std::make_pair(
            std::move(entryId), std::move(entryData)));
    }

    // Close the calibration file
    calibrationFile.close();

    // Check that the transformation from the velodyne to the rectified
    // camera coordinate is specified
    if (this->mCalibration.find("Tr") == this->mCalibration.end()) {
        ROS_ERROR("Calibration data should contain the transformation matrix "
                  "from the velodyne to the rectified camera coordinate");
        return false;
    }

    const std::vector<double>& camera0Velodyne = this->mCalibration["Tr"];

    // Check the number of values
    if (camera0Velodyne.size() != 12) {
        ROS_ERROR("Transformation matrix from the velodyne to the rectified "
                  "camera coordinate should contain exactly 12 elements");
        return false;
    }

    // Compute the velodyne to the rectified camera coordinate
    this->mTransformCamera0Velodyne.block<3, 4>(0, 0) =
        Eigen::Map<const Eigen::Matrix<double, 3, 4>>(camera0Velodyne.data());
    this->mTransformCamera0Velodyne.block<1, 4>(3, 0) =
        Eigen::Vector4d { 0.0, 0.0, 0.0, 1.0 };

    return true;
}

// Load the timestamps from the file
bool KittiOdometryDataset::LoadTimestamps()
{
    // Construct the file path to the timestamps
    const fs::path fileName = this->mSequencePath / "times.txt";

    // Check that the file exists
    if (!fs::is_regular_file(fileName)) {
        ROS_ERROR("Timestamp file does not exist: %s", fileName.c_str());
        return false;
    }

    // Open the timestamp file
    std::ifstream timestampFile { fileName };

    if (!timestampFile) {
        ROS_ERROR("Failed to open the timestamp file: %s", fileName.c_str());
        return false;
    }

    // Reset the timestamps
    this->mTimestamps.clear();

    // Read the timestamp file
    std::string timestampStr;
    while (std::getline(timestampFile, timestampStr))
        this->mTimestamps.push_back(std::stod(timestampStr));

    // Close the timestamp file
    timestampFile.close();

    return true;
}

// Load the ground-truth poses if available
bool KittiOdometryDataset::LoadGroundTruthPoses()
{
    // Construct the file path to the ground-truth poses
    const fs::path fileName = this->mPosePath / (this->mSequenceStr + ".txt");

    // Ground-truth poses are not available for sequences 11 to 21
    if (!fs::is_regular_file(fileName)) {
        ROS_INFO("Ground-truth poses are not available for the sequence %s",
                 this->mSequenceStr.c_str());
        return true;
    }

    // Open the ground-truth pose file
    std::ifstream groundTruthFile { fileName };

    if (!groundTruthFile) {
        ROS_ERROR("Failed to open the ground-truth pose file: %s",
                  fileName.c_str());
        return false;
    }

    // Read the ground-truth poses
    std::string groundTruthStr;

    while (std::getline(groundTruthFile, groundTruthStr)) {
        // Split the string with whitespaces
        std::istringstream strStream { groundTruthStr };
        std::vector<std::string> tokens;
        std::copy(std::istream_iterator<std::string>(strStream),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(tokens));

        // Skip the empty line
        if (tokens.empty())
            continue;

        // Check the number of values
        if (tokens.size() != 12) {
            ROS_WARN("Transformation matrix of the ground-truth should "
                     "contain exactly 12 elements");
            continue;
        }

        // Convert to the floating-point values
        std::vector<double> groundTruth;
        groundTruth.reserve(tokens.size());
        std::transform(tokens.begin(), tokens.end(),
                       std::back_inserter(groundTruth),
                       [](const std::string& str) { return std::stod(str); });

        // Convert to the transformation matrix
        Eigen::Matrix4d transformGroundTruth;
        transformGroundTruth.block<3, 4>(0, 0) =
            Eigen::Map<Eigen::Matrix<double, 3, 4>>(groundTruth.data());
        transformGroundTruth.block<1, 4>(3, 0) =
            Eigen::Vector4d { 0.0, 0.0, 0.0, 1.0 };
        this->mGroundTruthPoses.push_back(std::move(transformGroundTruth));
    }

    // Close the ground-truth pose file
    groundTruthFile.close();

    return true;
}

// Load the velodyne point cloud file names
bool KittiOdometryDataset::LoadPointCloudFileNames()
{
    // Find the velodyne point cloud files
    const fs::path pointCloudFilePath = this->mSequencePath / "velodyne";

    // Check that the directory exists
    if (!fs::is_directory(pointCloudFilePath)) {
        ROS_ERROR("Directory for the velodyne point clouds does not exist: %s",
                  pointCloudFilePath.c_str());
        return false;
    }

    // Find the velodyne point cloud files
    this->mPointCloudFileNames.clear();
    std::transform(fs::directory_iterator(pointCloudFilePath),
                   fs::directory_iterator(),
                   std::back_inserter(this->mPointCloudFileNames),
                   [](const fs::directory_entry& ent) { return ent.path(); });
    std::sort(this->mPointCloudFileNames.begin(),
              this->mPointCloudFileNames.end());

    return true;
}

} // namespace kitti
} // namespace loam
