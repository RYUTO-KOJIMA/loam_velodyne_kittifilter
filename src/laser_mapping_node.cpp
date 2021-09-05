
// laser_mapping_node.cpp

#include <ros/ros.h>
#include "loam_velodyne/LaserMapping.h"

/** Main node entry point. */
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle node;
    ros::NodeHandle privateNode("~");

    loam::LaserMapping laserMapping(0.1);

    if (laserMapping.setup(node, privateNode))
        laserMapping.spin();

    return 0;
}
