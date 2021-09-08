
// kitti_odometry_to_bag.cpp

#include <cstdlib>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "loam_velodyne/kitti/OdometryDataset.hpp"

namespace fs = std::experimental::filesystem;

using namespace loam::kitti;

int main(int argc, char** argv)
{
    if (argc != 6) {
        ROS_INFO("Usage: %s <Path to Kitti Odometry Dataset> "
                 "<Sequence> <Path to Output Bag File> "
                 "<Compression Type> <Topic Name>", argv[0]);
        return EXIT_FAILURE;
    }

    ros::Time::init();

    // Load the dataset path and the sequence number
    const std::string datasetPath = argv[1];
    const int datasetSequence = std::atoi(argv[2]);
    const std::string bagFileName = argv[3];
    const std::string compressionType = argv[4];
    const std::string topicName = argv[5];

    // Check the compression type
    if (compressionType != "none" &&
        compressionType != "bz2" &&
        compressionType != "lz4") {
        ROS_ERROR("Compression type should be `none`, `bz2`, or `lz4");
        return EXIT_FAILURE;
    }

    // Load the Kitti odometry dataset
    ROS_INFO("Loading the Kitti odometry dataset (sequence %d) from %s",
             datasetSequence, datasetPath.c_str());
    KittiOdometryDataset odometryDataset;
    if (!odometryDataset.Load(datasetPath, datasetSequence))
        return EXIT_FAILURE;

    // Open the ROS Bag
    ROS_INFO("Writing to a ROS Bag file %s", bagFileName.c_str());
    rosbag::Bag bagOut { bagFileName, rosbag::bagmode::Write };

    // Set the compression type
    if (compressionType == "none")
        bagOut.setCompression(rosbag::CompressionType::Uncompressed);
    else if (compressionType == "bz2")
        bagOut.setCompression(rosbag::CompressionType::BZ2);
    else if (compressionType == "lz4")
        bagOut.setCompression(rosbag::compression::LZ4);

    // Convert to the ROS Bag
    const std::size_t numOfData = odometryDataset.NumOfData();

    // Compute the timestamps of the velodyne point clouds
    const ros::Time startTime = ros::Time::now();
    auto addTime = [startTime](const double t) {
        return startTime + ros::Duration { t }; };
    std::vector<ros::Time> pointCloudTimestamps;
    pointCloudTimestamps.reserve(numOfData);
    std::transform(odometryDataset.Timestamps().begin(),
                   odometryDataset.Timestamps().end(),
                   std::back_inserter(pointCloudTimestamps),
                   addTime);

    for (std::size_t i = 0; i < numOfData; ++i) {
        // Load the velodyne point cloud from the file
        pcl::PointCloud<pcl::PointXYZI> pointCloud;
        if (!odometryDataset.PointCloudAt(i, pointCloud)) {
            ROS_ERROR("Failed to load the point cloud at index %zu", i);
            return EXIT_FAILURE;
        }

        if (i % 10 == 0)
            ROS_INFO("Processing the frame %zu ...", i);

        // Convert to the PointCloud2 message
        sensor_msgs::PointCloud2 pointCloudMsg;
        pcl::toROSMsg(pointCloud, pointCloudMsg);

        // Fill the header
        pointCloudMsg.header.seq = i;
        pointCloudMsg.header.stamp = pointCloudTimestamps.at(i);
        pointCloudMsg.header.frame_id = "velodyne";

        // Append to the ROS Bag file
        bagOut.write(topicName, pointCloudMsg.header.stamp, pointCloudMsg);
    }

    // Close the ROS Bag
    bagOut.close();

    return EXIT_SUCCESS;
}
