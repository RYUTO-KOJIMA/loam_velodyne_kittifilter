
// metrics_listener_node.cpp

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <ros/ros.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "loam_velodyne/ScanRegistrationMetrics.h"
#include "loam_velodyne/LaserOdometryMetrics.h"
#include "loam_velodyne/LaserMappingMetrics.h"

#include "loam_velodyne/SaveMetrics.h"

namespace pt = boost::property_tree;
using namespace loam_velodyne;

/* Convert the std::vector to std::string */
template <typename T>
std::string VecToString(const std::vector<T>& values,
                        const int precision = -1)
{
    std::ostringstream strStream;

    if (precision != -1)
        strStream << std::fixed << std::setprecision(precision);

    const std::size_t numOfValues = values.size();

    for (std::size_t i = 0; i < numOfValues; ++i)
        if (i == numOfValues - 1)
            strStream << values[i];
        else
            strStream << values[i] << ' ';

    return strStream.str();
}

/* Convert the std::vector<ros::Time> to std::string */
template <>
std::string VecToString(const std::vector<ros::Time>& values,
                        const int precision)
{
    std::ostringstream strStream;

    const std::size_t numOfValues = values.size();
    const std::streamsize width = strStream.width();
    const char fill = strStream.fill();

    for (std::size_t i = 0; i < numOfValues; ++i)
        if (i == numOfValues - 1)
            strStream << values[i].toNSec();
        else
            strStream << values[i].toNSec() << ' ';

    return strStream.str();
}

/* Convert the std::vector<ros::Duration> to std::string */
template <>
std::string VecToString(const std::vector<ros::Duration>& values,
                        const int precision)
{
    std::ostringstream strStream;

    const std::size_t numOfValues = values.size();
    const std::streamsize width = strStream.width();
    const char fill = strStream.fill();

    for (std::size_t i = 0; i < numOfValues; ++i)
        if (i == numOfValues - 1)
            strStream << values[i].toNSec();
        else
            strStream << values[i].toNSec() << ' ';

    return strStream.str();
}

class MetricsListener
{
public:
    /* Constructor */
    MetricsListener(ros::NodeHandle& node);
    /* Destructor */
    MetricsListener() = default;

    /* Handler for the metrics in ScanRegistration node */
    void OnScanRegistrationMetrics(
        const ScanRegistrationMetrics::ConstPtr& metrics);
    /* Handler for the metrics in LaserOdometry node */
    void OnLaserOdometryMetrics(
        const LaserOdometryMetrics::ConstPtr& metrics);
    /* Handler for the metrics in LaserMapping node */
    void OnLaserMappingMetrics(
        const LaserMappingMetrics::ConstPtr& metrics);

    /* Save the metrics in JSON format */
    bool SaveMetrics(SaveMetrics::Request& request,
                     SaveMetrics::Response& response);

private:
    /* Convert the ScanRegistration node metrics to string */
    pt::ptree ToPropertyTree(
        const ScanRegistrationMetrics::ConstPtr& metrics);
    /* Convert the LaserOdometry node metrics to string */
    pt::ptree ToPropertyTree(
        const LaserOdometryMetrics::ConstPtr& metrics);
    /* Convert the LaserMapping node metrics to string */
    pt::ptree ToPropertyTree(
        const LaserMappingMetrics::ConstPtr& metrics);

private:
    /* Subscriber for the metrics in ScanRegistration node */
    ros::Subscriber mSubScanRegistration;
    /* Subscriber for the metrics in LaserOdometry node */
    ros::Subscriber mSubLaserOdometry;
    /* Subscriber for the metrics in LaserMapping node */
    ros::Subscriber mSubLaserMapping;
    /* Service server for saving the results */
    ros::ServiceServer mSrvSaveMetrics;
    /* Vector of the metrics in ScanRegistration node */
    std::vector<ScanRegistrationMetrics::ConstPtr> mScanRegistrationMetrics;
    /* Vector of the metrics in LaserOdometry node */
    std::vector<LaserOdometryMetrics::ConstPtr> mLaserOdometryMetrics;
    /* Vector of the metrics in LaserMapping node */
    std::vector<LaserMappingMetrics::ConstPtr> mLaserMappingMetrics;
};

/* Constructor */
MetricsListener::MetricsListener(ros::NodeHandle& node)
{
    this->mSubScanRegistration = node.subscribe<ScanRegistrationMetrics>(
        "/scan_registration_metrics", 10,
        &MetricsListener::OnScanRegistrationMetrics, this);
    this->mSubLaserOdometry = node.subscribe<LaserOdometryMetrics>(
        "/laser_odometry_metrics", 10,
        &MetricsListener::OnLaserOdometryMetrics, this);
    this->mSubLaserMapping = node.subscribe<LaserMappingMetrics>(
        "/laser_mapping_metrics", 10,
        &MetricsListener::OnLaserMappingMetrics, this);
    this->mSrvSaveMetrics = node.advertiseService(
        "save_metrics", &MetricsListener::SaveMetrics, this);
}

/* Handler for the metrics in ScanRegistration node */
void MetricsListener::OnScanRegistrationMetrics(
    const ScanRegistrationMetrics::ConstPtr& metrics)
{
    this->mScanRegistrationMetrics.push_back(metrics);
}

/* Handler for the metrics in LaserOdometry node */
void MetricsListener::OnLaserOdometryMetrics(
    const LaserOdometryMetrics::ConstPtr& metrics)
{
    this->mLaserOdometryMetrics.push_back(metrics);
}

/* Handler for the metrics in LaserMapping node */
void MetricsListener::OnLaserMappingMetrics(
    const LaserMappingMetrics::ConstPtr& metrics)
{
    this->mLaserMappingMetrics.push_back(metrics);
}

/* Convert the ScanRegistration node metrics to string */
pt::ptree MetricsListener::ToPropertyTree(
    const ScanRegistrationMetrics::ConstPtr& metrics)
{
    pt::ptree jsonMetrics;

    jsonMetrics.put("Stamp", metrics->stamp.toNSec());

    jsonMetrics.put("PointExtractionTime",
                    metrics->point_extraction_time.toNSec());
    jsonMetrics.put("PointCloudStamp",
                    metrics->point_cloud_stamp.toNSec());
    jsonMetrics.put("NumOfUnprocessedPoints",
                    metrics->num_of_unprocessed_points);

    jsonMetrics.put("SweepStartStamp",
                    metrics->sweep_start_stamp.toNSec());
    jsonMetrics.put("NumOfFullResPoints",
                    metrics->num_of_full_res_points);
    jsonMetrics.put("NumOfPointsInRings",
                    VecToString(metrics->num_of_points_in_rings));

    jsonMetrics.put("FeatureExtractionTime",
                    metrics->feature_extraction_time.toNSec());
    jsonMetrics.put("NumOfSharpPoints",
                    metrics->num_of_sharp_points);
    jsonMetrics.put("NumOfLessSharpPoints",
                    metrics->num_of_less_sharp_points);
    jsonMetrics.put("NumOfFlatPoints",
                    metrics->num_of_flat_points);
    jsonMetrics.put("NumOfLessFlatPointsBeforeDS",
                    metrics->num_of_less_flat_points_before_ds);
    jsonMetrics.put("NumOfLessFlatPoints",
                    metrics->num_of_less_flat_points);

    return jsonMetrics;
}

/* Convert the LaserOdometry node metrics to string */
pt::ptree MetricsListener::ToPropertyTree(
    const LaserOdometryMetrics::ConstPtr& metrics)
{
    pt::ptree jsonMetrics;

    jsonMetrics.put("Stamp", metrics->stamp.toNSec());

    jsonMetrics.put("PointCloudStamp",
                    metrics->point_cloud_stamp.toNSec());
    jsonMetrics.put("NumOfFullResPoints",
                    metrics->num_of_full_res_points);
    jsonMetrics.put("NumOfLessSharpPoints",
                    metrics->num_of_less_sharp_points);
    jsonMetrics.put("NumOfLessFlatPoints",
                    metrics->num_of_less_flat_points);

    jsonMetrics.put("ProcessTime",
                    metrics->process_time.toNSec());
    jsonMetrics.put("NumOfQuerySharpPoints",
                    metrics->num_of_query_sharp_points);
    jsonMetrics.put("NumOfReferenceSharpPoints",
                    metrics->num_of_reference_sharp_points);
    jsonMetrics.put("NumOfQueryFlatPoints",
                    metrics->num_of_query_flat_points);
    jsonMetrics.put("NumOfReferenceFlatPoints",
                    metrics->num_of_reference_flat_points);

    jsonMetrics.put("OptimizationTime",
                    metrics->optimization_time.toNSec());
    jsonMetrics.put("NumOfIterations",
                    metrics->num_of_iterations);
    jsonMetrics.put("OptimizationIterationTimes",
                    VecToString(metrics->optimization_iteration_times));
    jsonMetrics.put("NumOfCorrespondences",
                    VecToString(metrics->num_of_correspondences));

    jsonMetrics.put("CornerCoefficientTimes",
                    VecToString(metrics->corner_coefficient_times));
    jsonMetrics.put("CornerCorrespondenceTimes",
                    VecToString(metrics->corner_correspondence_times));
    jsonMetrics.put("NumOfCornerCorrespondences",
                    VecToString(metrics->num_of_corner_correspondences));

    jsonMetrics.put("PlaneCoefficientTimes",
                    VecToString(metrics->plane_coefficient_times));
    jsonMetrics.put("PlaneCorrespondenceTimes",
                    VecToString(metrics->plane_correspondence_times));
    jsonMetrics.put("NumOfPlaneCorrespondences",
                    VecToString(metrics->num_of_plane_correspondences));

    return jsonMetrics;
}

/* Convert the LaserMapping node metrics to string */
pt::ptree MetricsListener::ToPropertyTree(
    const LaserMappingMetrics::ConstPtr& metrics)
{
    pt::ptree jsonMetrics;

    jsonMetrics.put("Stamp", metrics->stamp.toNSec());

    jsonMetrics.put("PointCloudStamp",
                    metrics->point_cloud_stamp.toNSec());
    jsonMetrics.put("NumOfFullResPoints",
                    metrics->num_of_full_res_points);
    jsonMetrics.put("NumOfSurroundPoints",
                    metrics->num_of_surround_points);
    jsonMetrics.put("NumOfSurroundPointsBeforeDS",
                    metrics->num_of_surround_points_before_ds);

    jsonMetrics.put("ProcessTime",
                    metrics->process_time.toNSec());
    jsonMetrics.put("NumOfQuerySharpPointsBeforeDS",
                    metrics->num_of_query_sharp_points_before_ds);
    jsonMetrics.put("NumOfQuerySharpPoints",
                    metrics->num_of_query_sharp_points);
    jsonMetrics.put("NumOfReferenceSharpPoints",
                    metrics->num_of_reference_sharp_points);
    jsonMetrics.put("NumOfQueryFlatPointsBeforeDS",
                    metrics->num_of_query_flat_points_before_ds);
    jsonMetrics.put("NumOfQueryFlatPoints",
                    metrics->num_of_query_flat_points);
    jsonMetrics.put("NumOfReferenceFlatPoints",
                    metrics->num_of_reference_flat_points);

    jsonMetrics.put("OptimizationTime",
                    metrics->optimization_time.toNSec());
    jsonMetrics.put("NumOfIterations",
                    metrics->num_of_iterations);
    jsonMetrics.put("OptimizationIterationTimes",
                    VecToString(metrics->optimization_iteration_times));
    jsonMetrics.put("NumOfCorrespondences",
                    VecToString(metrics->num_of_correspondences));

    jsonMetrics.put("CornerProcessTimes",
                    VecToString(metrics->corner_process_times));
    jsonMetrics.put("NumOfCornerCorrespondences",
                    VecToString(metrics->num_of_corner_correspondences));

    jsonMetrics.put("PlaneProcessTimes",
                    VecToString(metrics->plane_process_times));
    jsonMetrics.put("NumOfPlaneCorrespondences",
                    VecToString(metrics->num_of_plane_correspondences));

    return jsonMetrics;
}

/* Save the metrics in JSON format */
bool MetricsListener::SaveMetrics(SaveMetrics::Request& request,
                                  SaveMetrics::Response& response)
{
    /* Create Boost property tree from the results */
    pt::ptree jsonResults;
    pt::ptree jsonScanRegistrationMetrics;
    pt::ptree jsonLaserOdometryMetrics;
    pt::ptree jsonLaserMappingMetrics;

    for (const auto& metrics : this->mScanRegistrationMetrics)
        jsonScanRegistrationMetrics.add_child(
            "", this->ToPropertyTree(metrics));
    for (const auto& metrics : this->mLaserOdometryMetrics)
        jsonLaserOdometryMetrics.add_child(
            "", this->ToPropertyTree(metrics));
    for (const auto& metrics : this->mLaserMappingMetrics)
        jsonLaserMappingMetrics.add_child(
            "", this->ToPropertyTree(metrics));

    jsonResults.put("NumOfScanRegistrationMetrics",
                    this->mScanRegistrationMetrics.size());
    jsonResults.put("NumOfLaserOdometryMetrics",
                    this->mLaserOdometryMetrics.size());
    jsonResults.put("NumOfLaserMappingMetrics",
                    this->mLaserMappingMetrics.size());

    jsonResults.put_child("ScanRegistrationMetrics",
                          jsonScanRegistrationMetrics);
    jsonResults.put_child("LaserOdometryMetrics",
                          jsonLaserOdometryMetrics);
    jsonResults.put_child("LaserMappingMetrics",
                          jsonLaserMappingMetrics);

    /* Write the results in JSON format */
    try {
        pt::write_json(request.file_name, jsonResults);
        ROS_INFO("Metrics are saved to %s", request.file_name.c_str());
        response.result = true;
    } catch (const pt::json_parser_error& e) {
        ROS_ERROR("Failed to save the metrics to %s, message: %s",
                  request.file_name.c_str(), e.what());
        response.result = false;
    }

    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "metricsListener");
    ros::NodeHandle node;
    MetricsListener metricsListener { node };

    ros::spin();

    return EXIT_SUCCESS;
}
