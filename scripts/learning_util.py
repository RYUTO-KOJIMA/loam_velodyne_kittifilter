# coding: utf-8
# learning_util.py

import sys
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple, deque

import matplotlib
import matplotlib.pyplot as plt

from pointnet_model import PointNet_LSTM
from point_cloud_util import PointCloudXYZI

from abc import ABCMeta, abstractmethod

import torch_geometric
from torch_geometric.data import Data,Batch
from collections import deque
from nav_msgs.msg import Odometry

from data_util import fileprint


#GAMMA = 0.999
#EPS_START = 0.9
#EPS_END = 0.05
#EPS_DECAY = 200
#TARGET_UPDATE = 10

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    fileprint("A","device","cuda")
else:
    device = torch.device("cpu")
    fileprint("A","device","cpu")

# CONSTS
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#Transition2 = namedtuple('Transition2',
#                        ('pclA', 'longA' , 'shortA' , 'action', 'pclB', 'longB' , 'shortB' , 'reward'))

REPLAY_BUFFER_SIZE_LIMIT = 1000000
SAMPLE_LIMIT = 1000

OPTIMIZE_INTERVAL = 10
SEQ_LENGTH = 8
LEARN_BATCH_SIZE = 16
MIN_PCL_LIMIT_OF_START_LERNING = 100
TARGET_NET_UPDATE_INTERVAL = 10

GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 4000

EQ_PROB = 0.9

LEARNING_DEBUG = True
IS_VANILLA_LOAM = False # is vanilla loam

REWARD_ROT_RATIO = 10000
REWARD_TRANS_RATIO = 100

INPUT_LAST_CONVERTED_CLOUD_MAX = 5
LAST_CONVERTED_CLOUD_IDX_NORMALIZE_RATE = 100

fileprint("A","IS_VANILLA_LOAM",IS_VANILLA_LOAM)
fileprint("A","REPLAY_BUFFER_SIZE_LIMIT",REPLAY_BUFFER_SIZE_LIMIT)
fileprint("A","SAMPLE_LIMIT",SAMPLE_LIMIT)
fileprint("A","OPTIMIZE_INTERVAL",OPTIMIZE_INTERVAL)
fileprint("A","SEQ_LENGTH",SEQ_LENGTH)
fileprint("A","LEARN_BATCH_SIZE",LEARN_BATCH_SIZE)
fileprint("A","MIN_PCL_LIMIT_OF_START_LERNING",MIN_PCL_LIMIT_OF_START_LERNING)
fileprint("A","TARGET_NET_UPDATE_INTERVAL",TARGET_NET_UPDATE_INTERVAL)
fileprint("A","GAMMA",GAMMA)
fileprint("A","EPS_START",EPS_START)
fileprint("A","EPS_END",EPS_END)
fileprint("A","EPS_DECAY",EPS_DECAY)
fileprint("A","REWARD_ROT_RATIO",REWARD_ROT_RATIO)
fileprint("A","REWARD_TRANS_RATIO",REWARD_TRANS_RATIO)
fileprint("A","EQ_PROB",EQ_PROB)
fileprint("A","INPUT_LAST_CONVERTED_CLOUD_MAX",INPUT_LAST_CONVERTED_CLOUD_MAX)

# Do not Use
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemory_for_DRQN(object):

    def __init__(self, capacity = 100):

        self.capacity = capacity
        self.point_cloud = deque()
        self.mask = deque()
        self.reward = deque()

        # point cloud converted by mask and LOAM
        self.converted_pcl = []
    
    def check_capacity(self):
        while self.__len__() > self.capacity:
            self.point_cloud.popleft()
            self.mask.popleft()
            self.reward.popleft()

    def push_point_cloud(self, in_pcl):

        attach_idx_tensor = torch.full(( len(in_pcl) , 1 ),fill_value=0, dtype=torch.float32)
        in_pcl = torch.cat( (in_pcl , attach_idx_tensor) , dim=1 )
        for i in range( min(INPUT_LAST_CONVERTED_CLOUD_MAX , len(self.converted_pcl) ) ):
            conv_pcl = self.converted_pcl[-1-i].clone()
            attach_idx_tensor = torch.full(( len(conv_pcl) , 1 ),fill_value= (i+1)*LAST_CONVERTED_CLOUD_IDX_NORMALIZE_RATE , dtype=torch.float32)
            
            #print (conv_pcl.shape , attach_idx_tensor.shape)
            conv_pcl = torch.cat( (conv_pcl , attach_idx_tensor) , dim=1 )
            #print (in_pcl.shape,conv_pcl.shape)
            in_pcl = torch.cat( (in_pcl , conv_pcl) , dim=0 )

        self.point_cloud.append(in_pcl)
        self.check_capacity()
    
    def push_mask(self,in_mask):
        self.mask.append(in_mask)
        self.check_capacity()
    
    def push_reward(self,in_reward):
        self.reward.append(in_reward)
        self.check_capacity()
    
    def push_converted_pcl(self,in_pcl):
        self.converted_pcl.append(in_pcl)
    
    def get_last_point_cloud(self):
        if len(self.point_cloud) == 0:
            return None
        return self.point_cloud[-1]

    def get_last_mask(self):
        if len(self.mask) == 0:
            return None
        return self.mask[-1]

    def get_last_reward(self):
        if len(self.reward) == 0:
            return None
        return self.reward[-1]    

    def sample(self):

        # todo
        # sampled_???[batch , idx_in_sequence] = pushed_data
        sampled_pcls = [ [] for i in range(LEARN_BATCH_SIZE) ]
        sampled_masks = [ [] for i in range(LEARN_BATCH_SIZE) ]
        sampled_rewards = [ [] for i in range(LEARN_BATCH_SIZE) ]

        for batch_i in range(LEARN_BATCH_SIZE):

            seq_start_idx = random.randint(0,len(self)-SEQ_LENGTH)

            for i in range(seq_start_idx, seq_start_idx+SEQ_LENGTH):

                sampled_pcls[batch_i].append( self.point_cloud[i] )
                sampled_masks[batch_i].append( self.mask[i] )
                sampled_rewards[batch_i].append(self.reward[i])

        return sampled_pcls,sampled_masks,sampled_rewards

    def __len__(self):
        return min( len(self.point_cloud) , len(self.mask) , len(self.reward) )
    



# Agent Parent Class
class AgentDQN(object):
    
    def __init__(self,*network_args):
        self.replay_buffer = ReplayMemory_for_DRQN(REPLAY_BUFFER_SIZE_LIMIT)
        self.policy_net = None
        self.target_net = None
        self.init_networks(*network_args)
    
    @abstractmethod
    def init_networks(self , *network_args):
        # initialize policy_net & target_net
        pass
    
    @abstractmethod
    def get_mask(self, pointcloud):
        pass
    
    @abstractmethod
    def optimize_model(self, reward):
        pass


class AgentDQNPointNet(AgentDQN):

    def __init__(self, *network_args):
        
        #self.last_point_cloud = deque(maxlen=2)
        self.last_rot_error = 0
        self.last_trans_error = 0

        self.past_rot_errors = []
        self.past_trans_errors = []

        #self.last_mask = deque(maxlen=2)
        #self.last_long_memory_c  = deque(maxlen=2)
        #self.last_short_memory_h = deque(maxlen=2)
        #self.last_reward = None #deque(maxlen=2)
        
        self.num_processed_pcl = 0

        super(AgentDQNPointNet,self).__init__(*network_args)

    def init_networks(self, *network_args):
        self.class_num = network_args[0]
        self.ring_part_num = self.class_num.bit_length()-1
        self.policy_net = PointNet_LSTM(class_num=self.class_num , point_dimention=4)
        self.target_net = PointNet_LSTM(class_num=self.class_num , point_dimention=4)

        #self.policy_net.load_state_dict( torch.load("/home/kojima/saved_data/odometry_pomdp_loop1/" + "model_weight.pth") )

        self.target_net.load_state_dict( self.policy_net.state_dict() )
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters() , lr=0.001 , betas=(0.9, 0.999), eps=1e-08, weight_decay=0 ,amsgrad=False )

        #self.last_long_memory_c.append(self.policy_net.long_memory_c)
        #self.last_short_memory_h.append(self.policy_net.short_memory_h)

    def convert_inputcloud_to_tensor(self,given_point_cloud : PointCloudXYZI , sample_limit=float("inf")):
        # create Torch.tensor which contains both given_point_cloud and self.last_point_cloud
        # and random samples

        sample_pnum = min(sample_limit , given_point_cloud.get_point_num())
        sample_idxs = [i for i in range(given_point_cloud.get_point_num())]
        random.shuffle(sample_idxs)
        
        #pos = torch.tensor(
        #    [ [p.x,p.y,p.z] for p in given_point_cloud.points] 
        #    , dtype=torch.float32 , device=device )
        
        pos_list = [ [given_point_cloud.points[i].x , given_point_cloud.points[i].y , given_point_cloud.points[i].z] for i in sample_idxs[:sample_pnum] ]

        pos = torch.tensor(
            pos_list
            , dtype=torch.float32 , device=device )

        ret_data = Data( pos=pos )

        return ret_data

    def get_mask(self, pointcloud : PointCloudXYZI):

        converted_pointcloud = self.convert_inputcloud_to_tensor(pointcloud , SAMPLE_LIMIT)
        self.replay_buffer.push_point_cloud(converted_pointcloud.pos)
        converted_pointcloud = Data( pos=self.replay_buffer.get_last_point_cloud() )

        batch_pcl_indices = torch.zeros( converted_pointcloud.pos.shape[0] ,dtype=torch.long , device=device)
        wrapping_batch = MyBatch( batch_pcl_indices , converted_pointcloud.pos )

        #q_data,long_memory_c,short_memory_h = self.policy_net(wrapping_batch)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.num_processed_pcl / EPS_DECAY)
        
        with torch.no_grad():
            self.policy_net.eval() ; print ("set policy net to eval() mode in get_mask()")
            q_data = self.policy_net(wrapping_batch)
            print (q_data)
            q_max_idx = torch.argmax(q_data[0])

            if IS_VANILLA_LOAM:
                mask = self.class_num-1
                binary_mask = [ 1 if (2**i) & mask else 0 for i in range( self.ring_part_num ) ]
                self.replay_buffer.push_mask(mask)
            
            elif random.random() < EQ_PROB and len(self.replay_buffer.mask) != 0:
                last_mask = self.replay_buffer.get_last_mask()
                binary_mask = [ 1 if (2**i) & last_mask else 0 for i in range( self.ring_part_num ) ]
                self.replay_buffer.push_mask(last_mask)
                print ("mask is not changed")

            elif random.random() > eps_threshold and q_max_idx != 0:
                binary_mask = [ 1 if (2**i) & q_max_idx else 0 for i in range( self.ring_part_num ) ]
                self.replay_buffer.push_mask(q_max_idx)
                print ("mask from policy net")
            else:
                random_mask = random.randint(1,self.class_num-1)
                binary_mask = [ 1 if (2**i) & random_mask else 0 for i in range( self.ring_part_num ) ]
                self.replay_buffer.push_mask(random_mask)
                print ("mask from random generator")

            return binary_mask

    def optimize_model(self):

        if len(self.replay_buffer) < LEARN_BATCH_SIZE:
            return
        
        # sampled_???[batch , idx_in_sequence] = pushed_data
        sampled_pcls,sampled_masks,sampled_rewards = self.replay_buffer.sample()

        sampled_masks   = torch.tensor( [ [ sampled_masks[bi][-2] ] for bi in range(LEARN_BATCH_SIZE)] , dtype=torch.long , device=device )
        sampled_rewards = torch.tensor( [ sampled_rewards[bi][-2] for bi in range(LEARN_BATCH_SIZE)] , dtype=torch.float32 , device=device )
        sampled_masks.detach()
        sampled_rewards.detach()

        # save last policy_net hidden state and reset it
        saved_hidden_state = self.policy_net.get_now_hidden_state()
        self.policy_net.reset_hidden_state_zero()

        self.policy_net.train() ; print ("set policy net to train() mode in optimize_model()")

        # input sequence data
        q_datas = None
        for seq_idx in range(SEQ_LENGTH-1):
            
            # create batch
            concatinated_pcl = torch.zeros(0,4,dtype=torch.float32,device=device)  # point clouds concatinated to one tensor
            batch_pcl_indices = [] # batch_pcl_indices[i] = id of the batch which the i-th point belongs

            for batch_idx in range(LEARN_BATCH_SIZE):
                p = sampled_pcls[batch_idx][seq_idx]
                concatinated_pcl = torch.cat( (concatinated_pcl , p) , dim=0 )
                batch_pcl_indices += [batch_idx] * len(p)

            batch_pcl_indices = torch.tensor(batch_pcl_indices , dtype=torch.long , device=device)
            wrapping_batch = MyBatch( batch_pcl_indices , concatinated_pcl )

            q_data_policy = self.policy_net(wrapping_batch)
        
        print ("policy net calc end")

        # update network parameters
        last_hidden_state = self.policy_net.get_now_hidden_state()

        # input data to target net
        
        self.target_net.eval()
        self.target_net.set_hidden_state(last_hidden_state)
        self.target_net.eval()
        
        # create target net input data
        concatinated_pcl = torch.zeros(0,4,dtype=torch.float32,device=device)  # point clouds concatinated to one tensor
        batch_pcl_indices = [] # batch_pcl_indices[i] = id of the batch which the i-th point belongs
        for batch_idx in range(LEARN_BATCH_SIZE):
            p = sampled_pcls[batch_idx][SEQ_LENGTH-1]
            concatinated_pcl = torch.cat( (concatinated_pcl , p) , dim=0 )
            batch_pcl_indices += [batch_idx] * len(p)
        batch_pcl_indices = torch.tensor(batch_pcl_indices , dtype=torch.long , device=device)
        wrapping_batch = MyBatch( batch_pcl_indices , concatinated_pcl )
        # get Q(s',a') from target net
        with torch.no_grad():
            q_data_target = self.target_net(wrapping_batch)
            q_data_target.detach()
        
        print ("target net calc end")

        # calc loss
        q_max_target = torch.max(q_data_target , dim=1).values
        y = sampled_rewards + GAMMA * q_max_target

        # select q value based on actions
        # print (sampled_masks.size() , q_data_policy.size() , "sm:qdp")
        q_policy_based_on_action = torch.gather( input=q_data_policy , dim=1 , index=sampled_masks ).squeeze()
        # print (y.size() , q_policy_based_on_action.size() , "y:qpdoa")
        criterion = nn.SmoothL1Loss()
        loss = criterion( y , q_policy_based_on_action )
        print ("loss calc end")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        if LEARNING_DEBUG:
            print (self.num_processed_pcl , "th avarage training loss = " , torch.mean(loss).item() )
        fileprint("B",self.num_processed_pcl , "th_avarage_training_loss" , torch.mean(loss).item() )

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


        self.policy_net.eval() ; print ("set policy net to eval() mode in optimize_model()")
        # restore hidden state
        self.policy_net.set_hidden_state(saved_hidden_state)


        #batch_pcl_indices = torch.zeros( converted_pointcloud.pos.shape[0] ,dtype=torch.long , device=device)
        #wrapping_batch = MyBatch( batch_pcl_indices , converted_pointcloud.pos )



        #pclA_points = []
        #pclA_batch = []      

        #for i,transition in enumerate(transitions):
            
        #    pointcloudA = self.convert_inputcloud_to_tensor(transition.state[0])
            
        #    for p in pointcloudA:
        #        pclA_points.append( [p.x,p.y.p.z] )
        #        pclA_batch.append()
            

        #self.policy_net.eval()
        #batch_pcl_indices = torch.zeros( converted_pointcloud.pos.shape[0] ,dtype=torch.long , device=device)
        #wrapping_batch = MyBatch( batch_pcl_indices , converted_pointcloud.pos )

        #pclA_batch = []
        #return super().optimize_model(reward)

    def get_reward(self , reward):
        
        # push reward
        self.replay_buffer.push_reward(reward)
        self.num_processed_pcl += 1

        if len(self.replay_buffer) >= MIN_PCL_LIMIT_OF_START_LERNING  and  self.num_processed_pcl % OPTIMIZE_INTERVAL == 0:
            self.optimize_model()
        
        # policynet -> targetnet
        if self.num_processed_pcl % TARGET_NET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict( self.policy_net.state_dict() )
            self.target_net.eval()
            if LEARNING_DEBUG:
                print ("target net updated")
            
        # Add replay buffer
        #if len(self.last_point_cloud) == 2:
            
        #    stateA = (self.last_point_cloud[0] , self.last_long_memory_c[0] , self.last_short_memory_h[0])
        #    stateB = (self.last_point_cloud[1] , self.last_long_memory_c[1] , self.last_short_memory_h[1])
        #    actionA = self.last_mask[0]
        #    rewardA = self.last_reward
        #    self.replay_buffer.push( stateA , actionA , stateB , rewardA )

        #    # optimize model
        #    if len(self.replay_buffer) % OPTIMIZE_INTERVAL == 0:
        #        self.optimize_model()
        #self.last_reward = reward

    def calc_reward_old_v1(self , rot_error , trans_error):
        last_mask = list(map(int,format(self.replay_buffer.get_last_mask(),"b")))
        reduction_ratio = sum(last_mask) / len(last_mask)
        rot_error_diff   = abs(rot_error - self.last_rot_error)
        trans_error_diff = abs(trans_error - self.last_trans_error)

        self.last_rot_error = rot_error
        self.last_trans_error = trans_error
        return 1.0 / ( ( reduction_ratio * rot_error_diff * trans_error_diff ))
    
    def calc_reward_old_v2(self , rot_error , trans_error):

        mask_value = self.replay_buffer.get_last_mask()
        mask_bin = [ 1 if (mask_value & 2**i) > 0 else 0  for i in range(self.ring_part_num)]
        reduction_ratio = 1 - sum(mask_bin) / len(mask_bin)
        rot_error_abs   = abs(rot_error)
        trans_error_abs = abs(trans_error)

        print ("Reduction ratio:" , reduction_ratio)
        #reward = reduction_ratio - REWARD_ROT_RATIO * rot_error_abs - REWARD_TRANS_RATIO * trans_error_abs
        reward = 10 - REWARD_ROT_RATIO * rot_error_abs - REWARD_TRANS_RATIO * trans_error_abs

        self.last_rot_error = rot_error
        self.last_trans_error = trans_error

        return reward
    
    def calc_reward(self , rot_error , trans_error):
        
        LAST_LOOK_MAX = 100
        OK_LIMIT = 1.1

        look_num = min(LAST_LOOK_MAX , len(self.past_rot_errors))
        okcnt = 0
        for i in range(look_num):
            if abs(self.past_rot_errors[-1-i]) * OK_LIMIT > abs(rot_error):
                okcnt += 1
            if abs(self.past_trans_errors[-1-i]) * OK_LIMIT > abs(trans_error):
                okcnt += 1
        
        if look_num == 0:
            reward = 0
        else:
            reward = okcnt / (2 * look_num)

        self.past_rot_errors.append(rot_error)
        self.past_trans_errors.append(trans_error)

        self.last_rot_error = rot_error
        self.last_trans_error = trans_error
        return reward
    
    def get_converted_cloud(self, input_cloud : torch.tensor):
        self.replay_buffer.push_converted_pcl(input_cloud)

    

# my batch type (i can't understand torch_geometirc.data.Batch)
class MyBatch():
    def __init__(self , batch : torch.Tensor , pos : torch.Tensor):
        self.batch = batch
        self.pos = pos

