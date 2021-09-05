
// laser_odometry_node.cpp

#include <ros/ros.h>
#include "loam_velodyne/LaserOdometry.h"

/** Main node entry point. */
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle node;
    ros::NodeHandle privateNode("~");

    loam::LaserOdometry laserOdom(0.1);

    if (laserOdom.setup(node, privateNode))
        laserOdom.spin();

    return 0;
}
