import os
import math
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler

from model.SNN_model import define_net, define_optimizer, define_scheduler
from model.GCN import GraphNet, Attention

################
# Network Utils
################

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
            
##############################################################################
# Path + Omic
##############################################################################
#-----------------------------------------------------------------------------------------------#
# attention fusion
class AttentionFusion(nn.Module):
    def __init__(self, label_dim, dim1=256, dim2=256, hops = 3, hid=256, mmhid=128, dropout_rate=0.25, use_bilinear=1):
        super(AttentionFusion, self).__init__()
        self.use_bilinear = use_bilinear
        
        self.hops = hops
        self.Attention_1_1 = Attention(embed_dim=dim1, out_dim=hid)
        self.Attention_1_2 = Attention(embed_dim=dim2, out_dim=hid)
        self.Attention_2_1 = Attention(embed_dim=hid, out_dim=hid)
        self.Attention_2_2 = Attention(embed_dim=hid, out_dim=hid)
        
        self.linear_h1 = nn.Sequential(nn.Linear(hid, 128), nn.ReLU())
        self.linear_z1 = nn.Bilinear(hid, hid, 128) if use_bilinear else nn.Sequential(nn.Linear(hid+hid, 128))
        self.linear_o1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        self.linear_h2 = nn.Sequential(nn.Linear(hid, 128), nn.ReLU())
        self.linear_z2 = nn.Bilinear(hid, hid, 128) if use_bilinear else nn.Sequential(nn.Linear(hid+hid, 128))
        self.linear_o2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, mmhid),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, label_dim)
            )

    def forward(self, vec1, vec2):
        vec_1_1 = self.Attention_1_1(vec1, vec2)
        vec_1_2 = self.Attention_1_2(vec2, vec1)
        vec_2_1 = self.Attention_2_1(vec_1_1, vec_1_2)
        vec_2_2 = self.Attention_2_2(vec_1_2, vec_1_1)
        
        h1 = self.linear_h1(vec_2_1)
        #print('h1 ',h1.shape)
        z1 = self.linear_z1(vec_2_1, vec_2_2) if self.use_bilinear else self.linear_z1(torch.cat((vec_2_1, vec_2_2), dim=1))
        #print('z1 ',z1.shape)
        o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        
        #print('o1 ',o1.shape) 
        h2 = self.linear_h2(vec_2_2)
        z2 = self.linear_z2(vec_2_1, vec_2_2) if self.use_bilinear else self.linear_z2(torch.cat((vec_2_1, vec_2_2), dim=1))
        o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        
        vec_total = torch.cat((o1, o2), dim = 1)
        out = self.classifier(vec_total)
         
        return out
        
def define_attfusion(label_dim, dim1=256, dim2=256, mmhid=128, dropout_rate=0.25, gate1=1, gate2=1, addattention=1):
    fusion = AttentionFusion(label_dim=label_dim, dim1=dim1, dim2=dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    return fusion


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class PathomicFusionNet(nn.Module):
    def __init__(self, omic_loadname='', path_loadname='', label_dim=2, unimodal_frozen=True):
        super(PathomicFusionNet, self).__init__()
        
        self.omic_net = define_net()
        self.path_net = GraphNet(feature_length=128, nhid=256)
        
        if os.path.exists(omic_loadname):
            best_omic_ckpt = torch.load(omic_loadname, map_location=torch.device('cpu'))
            self.omic_net.load_state_dict(best_omic_ckpt['state_dict'])
            print("Loading Omic Models:\n", omic_loadname)
            if unimodal_frozen:
                dfs_freeze(self.omic_net)
            
        if os.path.exists(path_loadname):
            best_path_ckpt = torch.load(path_loadname, map_location=torch.device('cpu'))
            self.path_net.load_state_dict(best_path_ckpt['state_dict'])
            print("Loading Path Models:\n", path_loadname)
            if unimodal_frozen:
                dfs_freeze(self.path_net)
            
        self.fusion = define_attfusion(label_dim = label_dim, mmhid=128)
        
    def forward(self, data, omic, path_features, omic_features):
        _, path_vec, _ = self.path_net(data, omic_features, 1)
        _, omic_vec, _ = self.omic_net(path_features, omic, 1)
        
        out = self.fusion(path_vec, omic_vec)
        
        return out


