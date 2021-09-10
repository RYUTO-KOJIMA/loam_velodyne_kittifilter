
# LaserOdometryMetrics.msg

# Metrics updated in LaserOdometry::publishResult()
# Timestamp of the message
time stamp
# Timestamp of the sweep start
time point_cloud_stamp
# Number of the points in the full-resolution point cloud
uint32 num_of_full_res_points
# Number of the less-sharp points (including sharp points)
uint32 num_of_less_sharp_points
# Number of the less-flat points (including flat points)
uint32 num_of_less_flat_points

# Metrics updated in BasicLaserOdometry::process()
# Total processing time
duration process_time
# Number of the current sharp corner points (query points)
uint32 num_of_query_sharp_points
# Number of the previous less-sharp corner points (reference points)
uint32 num_of_reference_sharp_points
# Number of the current flat surface points (query points)
uint32 num_of_query_flat_points
# Number of the previous less-flat surface points (reference points)
uint32 num_of_reference_flat_points

# Metrics updated in BasicLaserOdometry::performOptimization()
# Processing time for the Gauss-Newton optimization
duration optimization_time
# Number of the Gauss-Newton iterations
uint32 num_of_iterations
# Processing time for the Gauss-Newton iteration
duration[] optimization_iteration_times
# Number of the correspondences
uint32[] num_of_correspondences

# Metrics updated in BasicLaserOdometry::computeCornerDistances() and
# BasicLaserOdometry::findCornerCorrespondence()
# Processing time for computing distances and coefficients for
# point-to-edge correspondences
duration[] corner_coefficient_times
# Processing time for finding point-to-edge correspondences
duration[] corner_correspondence_times
# Number of the point-to-edge correspondences
uint32[] num_of_corner_correspondences

# Metrics updated in BasicLaserOdometry::computePlaneDistances() and
# BasicLaserOdometry::findPlaneCorrespondence()
# Processing time for computing distances and coefficients for
# point-to-plane correspondences
duration[] plane_coefficient_times
# Processing time for finding point-to-plane correspondences
duration[] plane_correspondence_times
# Number of the point-to-plane correspondences
uint32[] num_of_plane_correspondences
