
# CMakeLists.txt

cmake_minimum_required(VERSION 3.1.3)
project(loam_velodyne)

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    sensor_msgs
    rosbag
    roscpp
    rospy
    std_msgs
    tf
    pcl_conversions
    message_generation)

find_package(Boost REQUIRED COMPONENTS system)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

add_message_files(
    FILES
    ScanRegistrationMetrics.msg
    LaserOdometryMetrics.msg
    LaserMappingMetrics.msg)

add_service_files(
    FILES
    SaveMetrics.srv
    SaveOdometry.srv)

generate_messages(
    DEPENDENCIES
    std_msgs)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${rosbag_INCLUDE_DIRS}
    #${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS})

catkin_package(
    CATKIN_DEPENDS geometry_msgs nav_msgs rosbag roscpp rospy
        std_msgs tf pcl_conversions
    DEPENDS EIGEN3 PCL
    INCLUDE_DIRS include
    LIBRARIES loam)

## Compile as C++14, supported in ROS Kinetic and newer
# set_property(TARGET invz_player PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/kitti)
add_subdirectory(src/lib)

add_executable(multiScanRegistration src/multi_scan_registration_node.cpp)
target_link_libraries(multiScanRegistration
    ${catkin_LIBRARIES} ${PCL_LIBRARIES} loam)

add_executable(laserOdometry src/laser_odometry_node.cpp)
target_link_libraries(laserOdometry
    ${catkin_LIBRARIES} ${PCL_LIBRARIES} loam)

add_executable(laserMapping src/laser_mapping_node.cpp)
target_link_libraries(laserMapping
    ${catkin_LIBRARIES} ${PCL_LIBRARIES} loam)

add_executable(transformMaintenance src/transform_maintenance_node.cpp)
target_link_libraries(transformMaintenance
    ${catkin_LIBRARIES} ${PCL_LIBRARIES} loam)



add_executable(kittiFilter src/kitti_filter_node.cpp)
target_link_libraries(kittiFilter ${catkin_LIBRARIES})
add_dependencies(kittiFilter loam_velodyne_gencpp)

add_executable(odometryListener src/odometry_listener_node.cpp)
target_link_libraries(odometryListener ${catkin_LIBRARIES} loam)
add_dependencies(odometryListener loam_velodyne_gencpp)

add_executable(metricsListener src/metrics_listener_node.cpp)
target_link_libraries(metricsListener ${catkin_LIBRARIES})
add_dependencies(metricsListener loam_velodyne_gencpp)

add_executable(kittiOdometry2Bag src/kitti_odometry_to_bag.cpp)
target_link_libraries(kittiOdometry2Bag
    ${catkin_LIBRARIES} loam_kitti stdc++fs)

if (CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    # TODO: Download test data
    catkin_download_test_data(${PROJECT_NAME}_test_data.tar.gz
        https://dl.dropboxusercontent.com/s/y4hn486461tfmpm/velodyne_loam_test_data.tar.gz
        MD5 3d5194e6981975588b7a93caebf79ba4)
    add_custom_target(${PROJECT_NAME}_test_data
        COMMAND ${CMAKE_COMMAND} -E tar -xzf velodyne_loam_test_data.tar.gz
        DEPENDS ${PROJECT_NAME}_test_data.tar.gz)
    configure_file(tests/loam.test.in
        ${PROJECT_BINARY_DIR}/test/loam.test)
    add_rostest(${PROJECT_BINARY_DIR}/test/loam.test
        DEPENDENCIES
        ${PROJECT_NAME}_test_data
        multiScanRegistration
        laserOdometry
        laserMapping
        transformMaintenance)
endif()
