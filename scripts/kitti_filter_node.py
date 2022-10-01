#!/home/kojima/.pyenv/shims/python
# -*- coding: utf-8 -*-
# license removed for brevity

import sys
import argparse
import rospy
import numpy as np
import pykitti

import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
import error_calc_util as ecu
from collections import deque
import time

DATASET_DIR = ""

SYNC_DEBUG = True
PUBLISH_WAIT = 0.01

lengths = [10.0 , 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]

class KittiFilter:

    def __init__(self):
        rospy.init_node('kittiFilter')
        self.pub_filterd_pcl = rospy.Publisher("/velodyne_points" , pc2.PointCloud2 , queue_size=1000)
        self.sub_raw_pcl = rospy.Subscriber("/raw_point_cloud",pc2.PointCloud2,self.callback_raw_pcl)
        self.sub_integrated_to_init = rospy.Subscriber("/integrated_to_init", Odometry , self.callback_integrated_to_init)

        self.mIntegratedTransforms = [] #result poses converted into rpy 
        self.poses_result = [] #result poses
        self.dists_result = [] #result distances

        dataset_dir = rospy.get_param("/kittiFilter/dataset_dir")
        sequence = rospy.get_param("/kittiFilter/sequence")
        self.dataset = pykitti.odometry(dataset_dir,str(sequence).zfill(2))
        self.poses_gt = self.dataset.poses #grandtruth poses
        self.dists_gt = ecu.trajectory_distances(self.poses_gt)

        # After receiving the results, pass new data to loam
        self.is_sync = False if str(rospy.get_param("/kittiFilter/sync")) in ("false","False","FALSE") else True
        self.get_pcl_queue = deque()
        self.num_pcl_loam_is_processing = 0
        self.lim_pcl_loam_is_processing = 1

        #print ("read gt from",dataset_dir, " sequence id=",str(sequence).zfill(2))

        #counters for debug
        self.cnt_get_raw_pcl = 0
        self.cnt_published_filtered_pcl = 0
        self.cnt_get_result = 0

    def callback_raw_pcl( self , gotmsg : pc2.PointCloud2 ) -> None:
        
        if SYNC_DEBUG:
            print ("receive /raw_point_cloud, No.",self.cnt_get_raw_pcl)
        self.cnt_get_raw_pcl += 1

        self.get_pcl_queue.append(gotmsg)

        if self.num_pcl_loam_is_processing < self.lim_pcl_loam_is_processing and self.get_pcl_queue:
            time.sleep(PUBLISH_WAIT)
            self.pub_filterd_pcl.publish(self.get_pcl_queue.popleft())
            self.num_pcl_loam_is_processing += 1
            
            if SYNC_DEBUG:
                print ("publish filtered_pcl(/velodyne_cloud), No.",self.cnt_published_filtered_pcl)
            self.cnt_published_filtered_pcl += 1

    def callback_integrated_to_init( self , data: Odometry ) -> None:
        
        if SYNC_DEBUG:
            print ("receive result, No.",self.cnt_get_result)
        self.cnt_get_result += 1

        self.num_pcl_loam_is_processing -= 1
        if self.num_pcl_loam_is_processing < self.lim_pcl_loam_is_processing and self.get_pcl_queue:
            time.sleep(PUBLISH_WAIT)
            self.pub_filterd_pcl.publish(self.get_pcl_queue.popleft())
            self.num_pcl_loam_is_processing += 1

            if SYNC_DEBUG:
                print ("publish filtered_pcl(/velodyne_cloud), No.",self.cnt_published_filtered_pcl)
            self.cnt_published_filtered_pcl += 1

        #self.poses_result.append(data)
        self.mIntegratedTransforms.append(
            ecu.ConvertToTwistTimed(data)
        )
    
        #process corresponding to json conversion and readout
        pose = self.mIntegratedTransforms[-1]

        # Create a 3x1 translation vector
        lo_t = np.array([ *pose.mTwist.pos ]) ; lo_t[0] *= -1 ; lo_t[1] *= -1
        # Create a 3x3 rotation matrix
        lo_Rx = pykitti.utils.rotx(-pose.mTwist.rot_x)
        lo_Ry = pykitti.utils.roty(-pose.mTwist.rot_y)
        lo_Rz = pykitti.utils.rotz(pose.mTwist.rot_z)
        lo_R = lo_Ry @ lo_Rx @ lo_Rz
        # Create a 4x4 transformation matrix
        lo_trans = pykitti.utils.transform_from_rot_trans(lo_R, lo_t)
            
        #calculate distance of nowpose to lastpose

        if len(self.poses_result) > 0:
            dist_nowflame_to_lastframe = ecu.trajectory_distance_pair(
                self.poses_result[-1], lo_trans)
            self.dists_result.append( dist_nowflame_to_lastframe )
            
        # Append the transformation
        self.poses_result.append(lo_trans)

        # get error of now frame
        odometry_errors = self.compute_sequence_error_now()

        #calc average rot & trans error

        if (len(odometry_errors) > 0):
            avr_rot_error = 0
            avr_trans_error = 0
            for e in odometry_errors:
                avr_rot_error += e.error_rot
                avr_trans_error += e.error_trans
            avr_rot_error /= len(odometry_errors)
            avr_trans_error /= len(odometry_errors)

            print ("CALCERROR:",avr_rot_error , ":" , avr_trans_error)
            
    def compute_sequence_error_now(self):

        step_size = 10

        now_frame = len(self.poses_result)-1

        errors = []

        for length in lengths:

            search_frame = None
            length_sum = 0
            for i in range(len(self.dists_result)-1 , -1 , -1):
                length_sum += self.dists_result[i]
                if length_sum > length:
                    search_frame = i
                    break
            
            if search_frame == None:
                continue
            
            #compute rotational and translational errors
            pose_delta_gt = np.linalg.inv(self.poses_gt[search_frame]) @ \
                            self.poses_gt[now_frame]
            pose_delta_result = np.linalg.inv(self.poses_result[search_frame]) @ \
                                self.poses_result[now_frame]
            pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt
            error_rot = ecu.rotation_error(pose_error)
            error_trans = ecu.translation_error(pose_error)

            ## Convert the number of frames to seconds
            num_frames = now_frame - search_frame + 1
            # Compute average speed
            speed = length / (0.1 * num_frames)

            # Collect error information
            now_error = ecu.Errors(search_frame,error_rot / length_sum,
                error_trans / length_sum , length_sum , speed)

            errors.append(now_error)
        
        return errors

if __name__ == '__main__':

    #if sys.argv:
    #    del sys.argv[1:]

    #perser = argparse.ArgumentParser()
    #perser.add_argument("--dataset_dir", required=True, type=str)
    #perser.add_argument("--sequence",required=True,type=str)
    #args = perser.parse_args()

    KittiFilter()
    rospy.spin()
