
/*only random filtering*/
#pragma once

#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>

/*
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <ros/node_handle.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "loam_velodyne/Common.h"
#include "loam_velodyne/Twist.h"

#include "loam_velodyne/SaveOdometry.h"
*/

// class definition
class KittiFilter
{
private:
    ros::Publisher pub_;
    ros::Subscriber sub_;
public:
    // constructor
    KittiFilter(ros::NodeHandle& node);
    KittiFilter() = default;
    // callback for pcl2
    void callback(const sensor_msgs::PointCloud2::ConstPtr& msg_sub);
};

//constructor
KittiFilter::KittiFilter(ros::NodeHandle& node)
{
    this->sub_ = node.subscribe<sensor_msgs::PointCloud2>("/raw_point_cloud" , 1000 ,
     &KittiFilter::callback, this);
    pub_ = node.advertise<sensor_msgs::PointCloud2> ("/filtered_point_cloud" , 1000);
}

void KittiFilter::callback(const sensor_msgs::PointCloud2::ConstPtr &msg_sub)
{
    sensor_msgs::PointCloud2::ConstPtr msg_pub;
    msg_pub = msg_sub;
    KittiFilter::pub_.publish(msg_pub);
}


int main(int argc, char** argv)
{

    ros::init(argc, argv, "kittiFilter");

    ros::NodeHandle node;
    KittiFilter kittiFilter(node);

    ros::spin();

}



