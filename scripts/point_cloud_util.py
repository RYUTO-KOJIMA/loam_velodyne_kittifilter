# coding: utf-8
# point_cloud_util.py

import math
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import random
import bisect
from torch_geometric.data import Data
import torch

import numpy as np
import rospy
import tf.transformations

from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry

from data_util import fileprint

# resolution of HDL 64E
# vertical   = 26.8 degree
# horizontal = 0.09 degree 
HDL64E_V_RESOLUTION = (31.5 / 63)  / 180 * math.pi
HDL64E_H_RESOLUTION = 0.09 / 180 * math.pi
#HDL64E_V_FOOTAGE = 

PC2_FIELDS = [ 
            PointField(name='x', offset=0, datatype=7, count=1),
            PointField(name='y', offset=4, datatype=7, count=1),
            PointField(name='z', offset=8, datatype=7, count=1),
            PointField(name='intensity', offset=16, datatype=7, count=1)
]


# Point cloud library (in Python) did not have a decent implementation

# my pointXYZI class

class PointXYZI():

    def __init__(self , x , y, z, intensity):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity
    
    def get_evation_angle(self):
        theta = math.asin(self.z / ( (self.x**2+self.y**2+self.z**2)**0.5 ) )
        return theta

# my point cloud class
class PointCloudXYZI():

    def __init__(self):
        self.points = []
        self.load_header = None
    
    # load point cloud from pc2_pointcloud2 instance
    def load_from_pc2_pointcloud2(self , given_pc2_point_cloud : pc2.PointCloud2):
        self.load_header = given_pc2_point_cloud.header
        points = pc2.read_points(given_pc2_point_cloud,skip_nans=True,field_names=("x", "y", "z" , "intensity"))
        for p in points:
            self.add_point( PointXYZI(p[0],p[1],p[2],p[3]) )
    
    # convert self.points to pc2_pointcloud2 and return
    def get_converted_pointcloud_to_pc2_pointcloud2(self) -> pc2.PointCloud2:
        
        if self.load_header == None:
            header = Header()
        else:
            header = self.load_header

        points = [ [p.x , p.y , p.z, p.intensity] for p in self.points ]

        return pc2.create_cloud(header,PC2_FIELDS,points)
    
    def get_converted_pointcloud_to_torch_geometric_data(self) -> Data:
        pos = torch.tensor( [ [p.x,p.y,p.z] for p in self.points] , dtype=torch.float32 )
        ret_data = Data( pos=pos )
        return ret_data

    # remove all points which satisfy func(point) == False
    def filter_points(self , func):
        passed_points = []
        for p in self.points:
            if func(p):
                passed_points.append(p)
        self.points = passed_points
    
    # The point cloud is equally divided into len(filter_list) based on the elevation angle,
    # and the i-th section from the smaller elevation angle is adopted only when filter_list[i] is True.
    def filter_pointcloud_by_evation_angle(self,filter_list):

        # pick points randomly and sort their evation angles
        MAX_RANDOM_PICK_RATE = 100
        random_pick_rate = min(MAX_RANDOM_PICK_RATE , len(self.points) // len(filter_list))
        random_pick_max = len(filter_list) * random_pick_rate - 1
        
        picked_evation_angles = []
        for _ in range(random_pick_max):
            pick_idx = random.randrange(0,len(self.points))
            picked_evation_angles.append( self.points[pick_idx].get_evation_angle() )
        picked_evation_angles.sort()

        boundary_evation_angles = picked_evation_angles[random_pick_rate::random_pick_rate]
        #print (len(boundary_evation_angles),"wowwww")

        passed_points = []
        for p in self.points:
            evation_angle_of_p = p.get_evation_angle()
            idx_in_bea = bisect.bisect_left(boundary_evation_angles , evation_angle_of_p)

            if filter_list[idx_in_bea]:
                passed_points.append(p)
        self.points = passed_points


    # get number of points
    def get_point_num(self):
        return len(self.points)

    # add a point
    def add_point(self , given_point:PointXYZI):
        self.points.append(given_point)
    
      

# calc scaline number of points in KittiDataset
# !!WARNING!! this functions is useless and theoretically uncorrectable
def get_scanline_kitti_from_xyz( point ) -> int:
    x,y,z = point.x , point.y , point.z
    theta = math.asin(z / ( (x**2+y**2+z**2)**0.5 ) )
    ring = theta / HDL64E_V_RESOLUTION
    return ring


# Using tf.transformations
def transform_pcl0(cloud: np.ndarray, odom_msg: Odometry):
    odom_pos: Point = odom_msg.pose.pose.position
    odom_quat: Quaternion = odom_msg.pose.pose.orientation

    # Get a rotation matrix and a translation vector
    rot = tf.transformations.quaternion_matrix([
        odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])
    rot = rot[:3, :3]
    trans = np.array([odom_pos.x, odom_pos.y, odom_pos.z], dtype=np.float64)

    cloud_transformed = cloud @ rot.T + trans
    return cloud_transformed

# Using SciPy
def transform_pcl1(cloud: np.ndarray, odom_msg: Odometry):
    odom_pos: Point = odom_msg.pose.pose.position
    odom_quat: Quaternion = odom_msg.pose.pose.orientation

    # Get a rotation matrix and a translation vector
    rot = Rotation.from_quat([
        odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])
    # Newer version of SciPy uses `as_matrix()` instead of `as_dcm()`
    rot = rot.as_dcm()
    trans = np.array([odom_pos.x, odom_pos.y, odom_pos.z], dtype=np.float64)

    cloud_transformed = cloud @ rot.T + trans
    return cloud_transformed