string family
int32 id
geometry_msgs/Point center      # centre in (x,y) pixel coordinates
geometry_msgs/Point[4] corners  # corners of tag ((x1,y1),(x2,y2),...)
geometry_msgs/PoseWithCovarianceStamped pose

# Normalized coordinates (0.0-1.0 range) for visualization compatibility
geometry_msgs/Point center_normalized      # centre in normalized coordinates
geometry_msgs/Point[4] corners_normalized  # corners in normalized coordinates
