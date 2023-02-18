#!/home/kojima/.pyenv/shims/python
# -*- coding: utf-8 -*-
# license removed for brevity

import sys
import argparse
import rospy
import numpy as np

#import numpy
#np.random.BitGenerator = numpy.random.bit_generator
import pykitti

import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
import error_calc_util as ecu
from collections import deque
import time

import tf.transformations

import point_cloud_util as pcu
import matplotlib.pyplot as plt
import learning_util

from data_util import fileprint
from data_util import SAVE_DATA_PATH

import os
import torch
import subprocess

import random

DATASET_DIR = ""

SYNC_DEBUG = True
ERROR_DEBUG = True
FILTERING_DEBUG = False
PUBLISH_WAIT = 0.01
RING_PART_NUM = 4

POINTNET_DQN_POINTNET_AND_LSTM = True

EXPORT_MODEL_FILENAME = "model_weight.pth"
ODOMETRY_FILENAME = "odometry.json"
EVALUATED_DATA_PREFIX = "gen"

fileprint("A","RING_PART_NUM",RING_PART_NUM)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lengths = [ 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0 ]
lengths2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]


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
        self.sequence_id = rospy.get_param("/kittiFilter/sequence")
        self.dataset = pykitti.odometry(dataset_dir,str(self.sequence_id).zfill(2))
        self.poses_gt = self.dataset.poses #grandtruth poses
        self.dists_gt = ecu.trajectory_distances(self.poses_gt)
        self.seq_length = len(self.poses_gt)

        # After receiving the results, pass new data to loam
        self.is_sync = False if str(rospy.get_param("/kittiFilter/sync")) in ("false","False","FALSE") else True
        self.get_pcl_queue = deque()
        self.num_pcl_loam_is_processing = 0
        self.lim_pcl_loam_is_processing = 1

        #print ("read gt from",dataset_dir, " sequence id=",str(sequence).zfill(2))

        # Learning objects
        self.agent = learning_util.AgentDQNPointNet(2**RING_PART_NUM)

        #counters for debug
        self.cnt_get_raw_pcl = 0
        self.cnt_published_filtered_pcl = 0
        self.cnt_get_result = 0

        # stock of masked point cloud
        self.last_published_pointcloud = None

        # print data
        fileprint("A","SEQUENCE_ID",self.sequence_id)
        fileprint("A","SEQ_LENGTH",self.seq_length)
        

    def publish_filterd_pointcloud(self):
        
        if len(self.get_pcl_queue) == 0:
            return 
        
        self.num_pcl_loam_is_processing += 1
        
        published_msg = self.get_pcl_queue.popleft()

        # test
        tmp_pointcloud = pcu.PointCloudXYZI()
        tmp_pointcloud.load_from_pc2_pointcloud2(published_msg)

        if POINTNET_DQN_POINTNET_AND_LSTM:
            # get_filter
            mask = self.agent.get_mask(tmp_pointcloud)
            #print (mask)
                
            # filtering
            tmp_pointcloud.filter_pointcloud_by_evation_angle(mask)

        if FILTERING_DEBUG:
            x = np.zeros( tmp_pointcloud.get_point_num() )
            for i,p in enumerate(tmp_pointcloud.points):
                r = pcu.get_scanline_kitti_from_xyz(p)
                x[i] = r
            y = np.random.rand( tmp_pointcloud.get_point_num() )
            plt.scatter(x,y)
            plt.savefig("onseeeeeeeeeeee.png")
            plt.show()
            
        #max_ring = float("-inf")
        #min_ring = float("inf")
        #for p in tmp_pointcloud.points:
        #    r = pcu.get_scanline_kitti(p)
        #    max_ring = max(max_ring , r)
        #    min_ring = min(min_ring , r)
        #print (min_ring,max_ring)

        self.last_published_pointcloud = tmp_pointcloud
        filtered_published_msg = tmp_pointcloud.get_converted_pointcloud_to_pc2_pointcloud2()

        time.sleep(PUBLISH_WAIT)
        self.pub_filterd_pcl.publish(filtered_published_msg)
            
        if SYNC_DEBUG:
            #print (mask)
            print ("publish filtered_pcl(/velodyne_cloud), No.",self.cnt_published_filtered_pcl , "mask=" , mask )

        fileprint ("C",self.cnt_published_filtered_pcl,"th_mask","".join(map(str,mask)))
        self.cnt_published_filtered_pcl += 1


    def callback_raw_pcl( self , gotmsg : pc2.PointCloud2 ) -> None:
        
        if SYNC_DEBUG:
            print ("receive /raw_point_cloud, No.",self.cnt_get_raw_pcl)
        self.cnt_get_raw_pcl += 1

        self.get_pcl_queue.append(gotmsg)

        if self.num_pcl_loam_is_processing == 0 and self.get_pcl_queue :
            self.publish_filterd_pointcloud()
           

    def callback_integrated_to_init( self , data: Odometry ) -> None:
        
        if SYNC_DEBUG:
            print ("receive result, No.",self.cnt_get_result)
        self.cnt_get_result += 1

        #self.poses_result.append(data)
        self.mIntegratedTransforms.append(
            ecu.ConvertToTwistTimed(data)
        )

        # calc converted_point_cloud
        sample_pnum = min(learning_util.SAMPLE_LIMIT , self.last_published_pointcloud.get_point_num())
        sample_idxs = [i for i in range(self.last_published_pointcloud.get_point_num())]
        random.shuffle(sample_idxs)
        pos_list = [ [self.last_published_pointcloud.points[i].x ,
                      self.last_published_pointcloud.points[i].y ,
                      self.last_published_pointcloud.points[i].z]
                      for i in sample_idxs[:sample_pnum] ]
        last_publishd_cloud_pos = np.array(pos_list , dtype=np.float32)
        converted_cloud_np = pcu.transform_pcl0(last_publishd_cloud_pos , data)
        converted_cloud_tc = torch.from_numpy(converted_cloud_np).clone()
        converted_cloud_tc = converted_cloud_tc.to(device=device,dtype=torch.float32)
        self.agent.get_converted_cloud(converted_cloud_tc)

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
        odometry_errors_long  = self.compute_sequence_error_now(lengths)
        odometry_errors       = self.compute_sequence_error_now(lengths2)

        #calc average rot & trans error

        if (len(odometry_errors) > 0):
            avr_rot_error = 0
            avr_trans_error = 0
            for e in odometry_errors:
                avr_rot_error += abs(e.error_rot)
                avr_trans_error += abs(e.error_trans)
            avr_rot_error /= len(odometry_errors)
            avr_trans_error /= len(odometry_errors)
        
            # train DQN
            if POINTNET_DQN_POINTNET_AND_LSTM:
                
                # get_reward
                reward = self.agent.calc_reward(avr_rot_error,avr_trans_error)
                
                if ERROR_DEBUG:
                    print ("CALCERROR(short):",avr_rot_error , ":" , avr_trans_error , " reward=", reward)
                fileprint("G",self.cnt_published_filtered_pcl-1,"th_avr_rot_error_short",avr_rot_error)
                fileprint("H",self.cnt_published_filtered_pcl-1,"th_avr_trans_error_short",avr_trans_error)
                fileprint("F",self.cnt_published_filtered_pcl-1,"th_reward",reward)

                # update
                self.agent.get_reward(reward)
        
        else:

            if POINTNET_DQN_POINTNET_AND_LSTM:
                reward = 0
                self.agent.get_reward(reward)
                if ERROR_DEBUG:
                    print ("Can't Calc Error : reward = 0")

                fileprint("G",self.cnt_published_filtered_pcl-1,"th_avr_rot_error_short",None)
                fileprint("H",self.cnt_published_filtered_pcl-1,"th_avr_trans_error_short",None)
                fileprint("F",self.cnt_published_filtered_pcl-1,"th_reward",0)

        # original kitti-evaluation
        if (len(odometry_errors_long) > 0):
            avr_rot_error_long = 0
            avr_trans_error_long = 0
            for e in odometry_errors_long:
                avr_rot_error_long += abs(e.error_rot)
                avr_trans_error_long += abs(e.error_trans)
            avr_rot_error_long   /= len(odometry_errors_long)
            avr_trans_error_long /= len(odometry_errors_long)
            fileprint("D",self.cnt_published_filtered_pcl-1,"th_avr_rot_error_long",avr_rot_error_long)
            fileprint("E",self.cnt_published_filtered_pcl-1,"th_avr_trans_error_long",avr_trans_error_long)
        else:
            fileprint("D",self.cnt_published_filtered_pcl-1,"th_avr_rot_error_long",None)
            fileprint("E",self.cnt_published_filtered_pcl-1,"th_avr_trans_error_long",None)


        

        self.num_pcl_loam_is_processing -= 1
        # publish next
        if self.num_pcl_loam_is_processing == 0 and self.get_pcl_queue:
            self.publish_filterd_pointcloud()
        
        if self.cnt_get_result == self.seq_length:
            print ("SEQUENCE COMPLETE")
            
            model_path = os.path.join( SAVE_DATA_PATH , EXPORT_MODEL_FILENAME )
            torch.save(self.agent.policy_net.state_dict(), model_path)
            print ("SAVED State-dict as", model_path )


            odometry_path = os.path.join( SAVE_DATA_PATH , ODOMETRY_FILENAME )
            command = "rosservice call /save_odometry " + odometry_path
            if os.system(command) == 0:
                print ("SAVED odometry as",odometry_path)
            else:
                print ("Odometry save failed")
            

            command = " ".join( [
                "python ~/catkin_ws/src/loam_velodyne/scripts/evaluate_kitti_odometry.py",
                " --dataset_dir /home/kojima/kitti-dataset/dataset",
                " --sequence " + str(self.sequence_id).zfill(2),
                " --odometry " + odometry_path,
                " --topic Integrated",
                " --out_dir " + SAVE_DATA_PATH,
                " --prefix " + EVALUATED_DATA_PREFIX
            ] )

            if os.system(command) == 0:
                print ("Genarated evaluated data" )
            else:
                print ("evaluated data generation faild")



    def compute_sequence_error_now(self , length_lis = lengths):

        step_size = 10

        now_frame = len(self.poses_result)-1

        errors = []

        for length in length_lis:

            search_frame = None
            length_sum = 0
            for i in range(now_frame , -1 , -1):
                if self.dists_gt[now_frame] - self.dists_gt[i] > length:
                    length_sum = self.dists_gt[now_frame] - self.dists_gt[i]
                    search_frame = i
                    break
            
            if search_frame == None:
                continue
            
            # pose_gt_s = self.poses_gt[search_frame]
            # pose_gt_n = self.poses_gt[now_frame]
            # pose_s = self.poses_result[search_frame]
            # pose_n = self.poses_result[now_frame]
            # to_str = lambda p: f"[{p[0][3]:.3f}, {p[1][3]:.3f}, {p[2][3]:.3f}]"

            # print(f"poses_gt[search_frame]: {to_str(pose_gt_s)}, " \
            #       f"poses_gt[now_frame]: {to_str(pose_gt_n)}, " \
            #       f"poses_result[search_frame]: {to_str(pose_s)}, "
            #       f"poses_result[now_frame]: {to_str(pose_n)}")

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
