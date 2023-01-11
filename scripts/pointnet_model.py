
import torch
from torch_geometric.nn import global_max_pool
import torch.nn as nn

CLASS_NUM = 5

class SymmFunction(nn.Module):
    def __init__(self):
        super(SymmFunction, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 512),
        )
        
    def forward(self, batch):
        x = self.shared_mlp(batch.pos)
        x = global_max_pool(x, batch.batch)
        return x



class InputTNet(nn.Module):
    def __init__(self):
        super(InputTNet, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 9)
        )
        
    def forward(self, x, batch):
        x = self.input_mlp(x)
        x = global_max_pool(x, batch)
        x = self.output_mlp(x)
        x = x.view(-1, 3, 3)
        id_matrix = torch.eye(3).to(x.device).view(1, 3, 3).repeat(x.shape[0], 1, 1)
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


class PointNet(nn.Module):
    def __init__(self,class_num=5):
        super(PointNet, self).__init__()
        self.input_tnet = InputTNet()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.feature_tnet = FeatureTNet()
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(256, class_num)
        )
        
    def forward(self, batch_data):
        x = batch_data.pos
        
        input_transform = self.input_tnet(x, batch_data.batch)
        transform = input_transform[batch_data.batch, :, :]
        x = torch.bmm(transform, x.view(-1, 3, 1)).view(-1, 3)
        
        x = self.mlp1(x)
        
        feature_transform = self.feature_tnet(x, batch_data.batch)
        transform = feature_transform[batch_data.batch, :, :]
        x = torch.bmm(transform, x.view(-1, 64, 1)).view(-1, 64)

        x = self.mlp2(x)        
        x = global_max_pool(x, batch_data.batch)
        x = self.mlp3(x)
        
        return x, input_transform, feature_transform