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

from pointnet_model import PointNet
from point_cloud_util import PointCloudXYZI

from abc import ABCMeta, abstractmethod

from torch_geometric.data import Data,Batch

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CONSTS
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

REPLAY_BUFFER_SIZE_LIMIT = 10000


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


# Agent Parent Class
class AgentDQN(object):
    
    def __init__(self,*network_args):
        self.replay_buffer = ReplayMemory(REPLAY_BUFFER_SIZE_LIMIT)
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
        super(AgentDQNPointNet,self).__init__(*network_args)

    def init_networks(self, *network_args):
        class_num = network_args[0]
        self.policy_net = PointNet(class_num=class_num)
        self.target_net = PointNet(class_num=class_num)

    def convert_inputcloud_to_tensor(self,given_point_cloud : PointCloudXYZI):
        return given_point_cloud.get_converted_pointcloud_to_torch_geometric_data()

    def get_mask(self, pointcloud : PointCloudXYZI):
        converted_pointcloud = self.convert_inputcloud_to_tensor(pointcloud)

        self.policy_net.eval()
        batch_pcl_indices = torch.zeros(pointcloud.get_point_num(),dtype=torch.long)
        wrapping_batch = MyBatch( batch_pcl_indices , converted_pointcloud.pos )
        mask,_,_ = self.policy_net(wrapping_batch)

        return mask

    def optimize_model(self, reward):
        return super().optimize_model(reward)

# my batch type (i can't understand torch_geometirc.data.Batch)
class MyBatch():
    def __init__(self , batch : torch.Tensor , pos : torch.Tensor):
        self.batch = batch
        self.pos = pos

