
// odometry_listener_node.cpp

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "loam_velodyne/Common.h"
#include "loam_velodyne/Twist.h"

#include "loam_velodyne/SaveOdometry.h"

struct TwistTimed
{
    /* Constructor */
    TwistTimed(const loam::Twist& twist, const ros::Time& timestamp,
               const std::string& frameId, const std::string& childFrameId) :
        mTwist(twist), mTimestamp(timestamp),
        mFrameId(frameId), mChildFrameId(childFrameId) { }

    /* Convert to the string representation */
    std::string ToString() const;

    /* Twist (rotation in Euler angles and position) */
    loam::Twist mTwist;
    /* Timestamp */
    ros::Time   mTimestamp;
    /* Frame Id */
    std::string mFrameId;
    /* Child frame Id */
    std::string mChildFrameId;
};

/* Convert to the string representation */
std::string TwistTimed::ToString() const
{
    std::stringstream strStream;

    strStream << this->mTimestamp.toNSec() << ' '
              << this->mTwist.pos.x() << ' '
              << this->mTwist.pos.y() << ' '
              << this->mTwist.pos.z() << ' '
              << this->mTwist.rot_x.rad() << ' '
              << this->mTwist.rot_y.rad() << ' '
              << this->mTwist.rot_z.rad();
    return strStream.str();
}

class OdometryListener
{
public:
    /* Constructor */
    OdometryListener(ros::NodeHandle& node);
    /* Destructor */
    OdometryListener() = default;

    /* Handler for the odometry results */
    void OnOdomToInit(
        const nav_msgs::Odometry::ConstPtr& odomToInit);
    /* Handler for the mapping results */
    void OnMappedToInit(
        const nav_msgs::Odometry::ConstPtr& mappedToInit);
    /* Handler for the integrated results */
    void OnIntegratedToInit(
        const nav_msgs::Odometry::ConstPtr& integratedToInit);

    /* Save the results in JSON format */
    bool SaveResults(loam_velodyne::SaveOdometry::Request& request,
                     loam_velodyne::SaveOdometry::Response& response);

private:
    /* Convert to the quaternion in ROS to the Euler angles in LOAM */
    void ConvertQuaternionToRPY(const geometry_msgs::Quaternion& rotationQuat,
                                float& roll, float& pitch, float& yaw) const;
    /* Convert to the internal transform representation */
    TwistTimed ConvertToTwistTimed(
        const nav_msgs::Odometry::ConstPtr& odometryMsg) const;

private:
    /* Subscriber for the odometry results */
    ros::Subscriber mSubOdomToInit;
    /* Subscriber for the mapping results */
    ros::Subscriber mSubMappedToInit;
    /* Subscriber for the integrated results */
    ros::Subscriber mSubIntegratedToInit;
    /* Service server for saving the results */
    ros::ServiceServer mSrvResults;
    /* Vector of the odometry results */
    std::vector<TwistTimed> mOdometryTransforms;
    /* Vector of the mapping results */
    std::vector<TwistTimed> mMappingTransforms;
    /* Vector of the integrated results */
    std::vector<TwistTimed> mIntegratedTransforms;
};

/* Constructor */
OdometryListener::OdometryListener(ros::NodeHandle& node)
{
    this->mSubOdomToInit = node.subscribe<nav_msgs::Odometry>(
        "/laser_odom_to_init", 10,
        &OdometryListener::OnOdomToInit, this);
    this->mSubMappedToInit = node.subscribe<nav_msgs::Odometry>(
        "/aft_mapped_to_init", 10,
        &OdometryListener::OnMappedToInit, this);
    this->mSubIntegratedToInit = node.subscribe<nav_msgs::Odometry>(
        "/integrated_to_init", 10,
        &OdometryListener::OnIntegratedToInit, this);
    this->mSrvResults = node.advertiseService(
        "save_odometry", &OdometryListener::SaveResults, this);
}

/* Handler for the odometry results */
void OdometryListener::OnOdomToInit(
    const nav_msgs::Odometry::ConstPtr& odomToInit)
{
    this->mOdometryTransforms.push_back(
        this->ConvertToTwistTimed(odomToInit));
}

/* Handler for the mapping results */
void OdometryListener::OnMappedToInit(
    const nav_msgs::Odometry::ConstPtr& mappedToInit)
{
    this->mMappingTransforms.push_back(
        this->ConvertToTwistTimed(mappedToInit));
}

/* Handler for the integrated results */
void OdometryListener::OnIntegratedToInit(
    const nav_msgs::Odometry::ConstPtr& integratedToInit)
{
    this->mIntegratedTransforms.push_back(
        this->ConvertToTwistTimed(integratedToInit));
}

/* Save the results in JSON format */
bool OdometryListener::SaveResults(
    loam_velodyne::SaveOdometry::Request& request,
    loam_velodyne::SaveOdometry::Response& response)
{
    namespace pt = boost::property_tree;

    auto checkFrameId = [](const std::vector<TwistTimed>& transforms) {
        if (transforms.empty())
            return true;

        const std::string& frameId = transforms.front().mFrameId;
        const std::string& childFrameId = transforms.front().mChildFrameId;

        return std::all_of(transforms.begin(), transforms.end(),
            [frameId, childFrameId](const TwistTimed& transform) {
                return transform.mFrameId == frameId &&
                       transform.mChildFrameId == childFrameId; });
    };

    /* Make sure that the frame Id is consistent */
    if (!checkFrameId(this->mOdometryTransforms) ||
        !checkFrameId(this->mMappingTransforms) ||
        !checkFrameId(this->mIntegratedTransforms)) {
        response.result = false;
        return true;
    }

    /* Create Boost property tree from the results */
    pt::ptree jsonResults;
    pt::ptree odometryResults;
    pt::ptree mappingResults;
    pt::ptree integratedResults;

    for (const auto& transformOdom : this->mOdometryTransforms)
        odometryResults.add("", transformOdom.ToString());

    for (const auto& transformMapped : this->mMappingTransforms)
        mappingResults.add("", transformMapped.ToString());

    for (const auto& transformIntegrated : this->mIntegratedTransforms)
        integratedResults.add("", transformIntegrated.ToString());

    if (!this->mOdometryTransforms.empty()) {
        jsonResults.put("Odometry.FrameId",
                        this->mOdometryTransforms.front().mFrameId);
        jsonResults.put("Odometry.ChildFrameId",
                        this->mOdometryTransforms.front().mChildFrameId);
        jsonResults.put("Odometry.NumOfResults",
                        this->mOdometryTransforms.size());
        jsonResults.put_child("Odometry.Results", odometryResults);
    }

    if (!this->mMappingTransforms.empty()) {
        jsonResults.put("Mapping.FrameId",
                        this->mMappingTransforms.front().mFrameId);
        jsonResults.put("Mapping.ChildFrameId",
                        this->mMappingTransforms.front().mChildFrameId);
        jsonResults.put("Mapping.NumOfResults",
                        this->mMappingTransforms.size());
        jsonResults.put_child("Mapping.Results", mappingResults);
    }

    if (!this->mIntegratedTransforms.empty()) {
        jsonResults.put("Integrated.FrameId",
                        this->mIntegratedTransforms.front().mFrameId);
        jsonResults.put("Integrated.ChildFrameId",
                        this->mIntegratedTransforms.front().mChildFrameId);
        jsonResults.put("Integrated.NumOfResults",
                        this->mIntegratedTransforms.size());
        jsonResults.put_child("Integrated.Results", integratedResults);
    }

    /* Write the results in JSON format */
    try {
        pt::write_json(request.file_name, jsonResults);
        ROS_INFO("Odometry is saved to %s", request.file_name.c_str());
        response.result = true;
    } catch (const pt::json_parser_error& e) {
        ROS_ERROR("Failed to save the odometry to %s, message: %s",
                  request.file_name.c_str(), e.what());
        response.result = false;
    }

    return true;
}

/* Convert to the quaternion in ROS to the Euler angles in LOAM */
void OdometryListener::ConvertQuaternionToRPY(
    const geometry_msgs::Quaternion& rotationQuat,
    float& roll, float& pitch, float& yaw) const
{
    const tf::Quaternion swappedQuat {
        rotationQuat.z, -rotationQuat.x, -rotationQuat.y, rotationQuat.w };
    const tf::Matrix3x3 rotationMat { swappedQuat };

    double swappedRoll;
    double swappedPitch;
    double swappedYaw;
    rotationMat.getRPY(swappedRoll, swappedPitch, swappedYaw);

    roll = -static_cast<float>(swappedPitch);
    pitch = -static_cast<float>(swappedYaw);
    yaw = static_cast<float>(swappedRoll);
}

/* Convert to the internal transform representation */
TwistTimed OdometryListener::ConvertToTwistTimed(
    const nav_msgs::Odometry::ConstPtr& odometryMsg) const
{
    float roll;
    float pitch;
    float yaw;
    this->ConvertQuaternionToRPY(
        odometryMsg->pose.pose.orientation, roll, pitch, yaw);

    const auto odomPosition = odometryMsg->pose.pose.position;
    loam::Twist transformOdom {
        static_cast<float>(odomPosition.x),
        static_cast<float>(odomPosition.y),
        static_cast<float>(odomPosition.z),
        roll, pitch, yaw };
    TwistTimed transformTimed {
        transformOdom, odometryMsg->header.stamp,
        odometryMsg->header.frame_id, odometryMsg->child_frame_id };

    return transformTimed;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "odometryListener");
    ros::NodeHandle node;
    OdometryListener odometryListener { node };

    ros::spin();

    return EXIT_SUCCESS;
}
