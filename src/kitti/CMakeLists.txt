
# CMakeLists.txt

add_library(loam_kitti
    OdometryDataset.cpp)

target_link_libraries(loam_kitti ${catkin_LIBRARIES} ${PCL_LIBRARIES})
