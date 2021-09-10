
# LaserMappingMetrics.msg

# Metrics updated in LaserMapping::publishResult()
# Timestamp of the message
time stamp
# Timestamp of the sweep start
time point_cloud_stamp
# Number of the points in the full-resolution point cloud
uint32 num_of_full_res_points
# Number of the points in the map of the surrounding
uint32 num_of_surround_points
# Number of the points in the map of the surrounding (before downsampling)
uint32 num_of_surround_points_before_ds

# Metrics updated in BasicLaserMapping::process()
# Total processing time
duration process_time
# Number of the current less-sharp corner points (before downsampling)
uint32 num_of_query_sharp_points_before_ds
# Number of the current less-sharp corner points (query points)
uint32 num_of_query_sharp_points
# Number of the corner points in the map (reference points)
uint32 num_of_reference_sharp_points
# Number of the current less-flat surface points (before downsampling)
uint32 num_of_query_flat_points_before_ds
# Number of the current less-flat surface points (query points)
uint32 num_of_query_flat_points
# Number of the flat surface points in the map (reference points)
uint32 num_of_reference_flat_points

# Metrics updated in BasicLaserMapping::optimizeTransformTobeMapped()
# Processing time for the Gauss-Newton optimization
duration optimization_time
# Number of the Gauss-Newton iterations
uint32 num_of_iterations
# Processing time for the Gauss-Newton iteration
duration[] optimization_iteration_times
# Number of the correspondences
uint32[] num_of_correspondences

# Metrics updated in BasicLaserMapping::computeCornerDistances()
# Processing time for finding point-to-edge correspondences and
# computing distances and coefficients
duration[] corner_process_times
# Number of the point-to-edge correspondences
uint32[] num_of_corner_correspondences

# Metrics updated in BasicLaserMapping::computePlaneDistances()
# Processing time for finding point-to-plane correspondences and
# computing distances and coefficients
duration[] plane_process_times
# Number of the point-to-plane correspondences
uint32[] num_of_plane_correspondences
