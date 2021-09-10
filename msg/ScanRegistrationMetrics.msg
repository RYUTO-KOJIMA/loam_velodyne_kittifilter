
# ScanRegistrationMetrics.msg

# Metrics updated in ScanRegistration::publishResult()
# Timestamp of the message
time stamp

# Metrics updated in MultiScanRegistration
# Processing time for extracting the valid points
duration point_extraction_time
# Timestamp of the input point cloud
time point_cloud_stamp
# Number of the points in the input point cloud
uint32 num_of_unprocessed_points

# Metrics updated in BasicScanRegistration::processScanlines()
# Timestamp of the sweep start
time sweep_start_stamp
# Number of the points in the full-resolution point cloud
uint32 num_of_full_res_points
# Number of the points in each scan (ring)
uint32[] num_of_points_in_rings

# Metrics updated in BasicScanRegistration::extractFeatures()
# Processing time for the feature extraction
duration feature_extraction_time
# Number of the sharp points
uint32 num_of_sharp_points
# Number of the less-sharp points (including sharp points)
uint32 num_of_less_sharp_points
# Number of the flat points
uint32 num_of_flat_points
# Number of the less-flat points (before downsampling, including flat points)
uint32 num_of_less_flat_points_before_ds
# Number of the less-flat points (after downsampling, including flat points)
uint32 num_of_less_flat_points
