
import torch
from torch_geometric.nn import global_max_pool
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from data_util import fileprint


CLASS_NUM = 5
BATCH_SIZE = 1 #change later

class SymmFunction(nn.Module):
    def __init__(self,point_dimention=3):
        super(SymmFunction, self).__init__()
        self.point_dimention = point_dimention
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.point_dimention, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 512),
        )
        
    def forward(self, batch):
        x = self.shared_mlp(batch.pos)
        x = global_max_pool(x, batch.batch)
        return x



class InputTNet(nn.Module):
    def __init__(self,point_dimention=3):
        super(InputTNet, self).__init__()
        self.point_dimention = point_dimention
        self.input_mlp = nn.Sequential(
            nn.Linear(self.point_dimention, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, self.point_dimention ** 2)
        )
        
    def forward(self, x, batch):
        x = self.input_mlp(x)
        x = global_max_pool(x, batch)
        x = self.output_mlp(x)
        x = x.view(-1, self.point_dimention, self.point_dimention)
        id_matrix = torch.eye(self.point_dimention).to(x.device).view(1, self.point_dimention, self.point_dimention).repeat(x.shape[0], 1, 1)
        x = id_matrix + x
        return x



class FeatureTNet(nn.Module):
    def __init__(self):
        super(FeatureTNet, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 64*64)
        )
        
    def forward(self, x, batch):
        x = self.input_mlp(x)
        x = global_max_pool(x, batch)
        x = self.output_mlp(x)
        x = x.view(-1, 64, 64)
        id_matrix = torch.eye(64).to(x.device).view(1, 64, 64).repeat(x.shape[0], 1, 1)
        x = id_matrix + x
        return x


class GraphConvLayer(nn.Module):
    
    def __init__(self, class_num=1):
        super(GraphConvLayer,self).__init__()
        self.conv1 = GCNConv(1,64)
        self.conv2 = GCNConv(64,1)

    def forward(self,x,edge_index):
        x = self.conv1(x , edge_index)
        x = self.conv2(x , edge_index)
        return x


class PointNet_LSTM(nn.Module):
    def __init__(self,class_num=1024,point_dimention=3):
        super(PointNet_LSTM, self).__init__()

        self.point_dimention = point_dimention
        self.input_tnet = InputTNet(self.point_dimention)
        self.mlp1 = nn.Sequential(
            nn.Linear(point_dimention, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.feature_tnet = FeatureTNet()
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        )

        #self.lstm1 = nn.LSTM(input_size = 1024 , hidden_size = 1024 , num_layers = 5)
        self.lstm1 = nn.LSTMCell(input_size=1024 , hidden_size=1024)
        self.hc_tuple_of_lstm1 = None

        #self.long_memory_c   = torch.randn( self.lstm1.num_layers , BATCH_SIZE , self.lstm1.input_size ) * 0.01
        #self.short_memory_h  = torch.randn( self.lstm1.num_layers , BATCH_SIZE , self.lstm1.input_size ) * 0.01

        #self.long_memory_c   = None #torch.zeros( self.lstm1.num_layers  , self.lstm1.input_size )
        #self.short_memory_h  = None #torch.zeros( self.lstm1.num_layers  , self.lstm1.input_size )
        #self.long_memory_c.to("cuda")
        #self.short_memory_h.to("cuda")


        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(256, class_num)
        )

        self.gconv = GraphConvLayer()
        self.class_num = class_num

    def reset_hidden_state_zero(self):
        self.hc_tuple_of_lstm1 = None
    
    def set_hidden_state(self, in_hidden_state):
        self.hc_tuple_of_lstm1 = in_hidden_state

    def get_now_hidden_state(self):
        return self.hc_tuple_of_lstm1
        
    def forward(self, batch_data):
        x = batch_data.pos
        
        input_transform = self.input_tnet(x, batch_data.batch)
        transform = input_transform[batch_data.batch, :, :]
        x = torch.bmm(transform, x.view(-1, self.point_dimention, 1)).view(-1, self.point_dimention)
        
        x = self.mlp1(x)
        
        feature_transform = self.feature_tnet(x, batch_data.batch)
        transform = feature_transform[batch_data.batch, :, :]
        x = torch.bmm(transform, x.view(-1, 64, 1)).view(-1, 64)

        x = self.mlp2(x)        
        x = global_max_pool(x, batch_data.batch)

        #print (x.shape)

        #x,(self.long_memory_c, self.short_memory_h) = self.lstm1( x , (self.long_memory_c,self.short_memory_h) )

        #print (self.hc_tuple_of_lstm1)
        if self.hc_tuple_of_lstm1 != None:
            self.hc_tuple_of_lstm1 = ( self.hc_tuple_of_lstm1[0].detach() , self.hc_tuple_of_lstm1[1].detach() )
        self.hc_tuple_of_lstm1 = self.lstm1( x , self.hc_tuple_of_lstm1 )

        x = self.mlp3(self.hc_tuple_of_lstm1[0])

        batch_size = len(x)
        x = torch.reshape(x , (1,-1))
        x = torch.transpose(x,0,1)

        pow_of_2 = set( [2**i for i in range(self.class_num.bit_length())] )
        edge = [ [] , [] ]
        for i in range(len(x)):
            for bit in range(self.class_num.bit_length()-1):
                j = i ^ (2**bit)
                edge[0].append(i)
                edge[1].append(j)
        edge_index = torch.tensor( edge , dtype=torch.long )

        x = self.gconv(x , edge_index)

        x = torch.transpose(x,0,1)
        x = torch.reshape(x , (batch_size,-1))
        
        return x
        #return x,None,None # self.long_memory_c , self.short_memory_h

