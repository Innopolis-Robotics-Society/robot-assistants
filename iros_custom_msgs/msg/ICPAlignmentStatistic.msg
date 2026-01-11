std_msgs/Header header

# Reference
string ref_mesh_path
float32 ref_mesh_scale

# Processing
float32 voxel_size
float32 size_ratio_scan_over_ref
bool size_mismatch_warning

# ICP quality
float32 icp_fitness
float32 icp_rmse

# Distances (scan -> ref)
float32 scan_to_ref_mean
float32 scan_to_ref_p95
float32 scan_to_ref_max
float32 scan_to_ref_over_thresh_ratio

# Distances (ref -> scan)
float32 ref_to_scan_mean
float32 ref_to_scan_p95
float32 ref_to_scan_max
float32 ref_to_scan_over_thresh_ratio

# Threshold used to compute over_thresh_ratio
float32 dist_thresh

# Counts used
uint32 scan_points
uint32 ref_points

# Status
bool success
string message
