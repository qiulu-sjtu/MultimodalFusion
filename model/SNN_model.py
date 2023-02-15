import os
import math
import numpy as np

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

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.w_kx.data.uniform_(-stdv, stdv)
        self.w_qx.data.uniform_(-stdv, stdv)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=0)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=0)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head, ?*k_len, embed_dim) -> (n_head*?, k_len, hidden_dim)
        # qx: (n_head, ?*q_len, embed_dim) -> (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*q_len, embed_dim)
        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim)  # (n_head*?, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim)  # (n_head*?, q_len, hidden_dim)
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1) #(1,256,10)
            qkt = torch.bmm(qx, kt)  #(1,304,10)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1) #(1,304,10)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        #output = output + k
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        if len(output.shape) == 3:
            output = torch.squeeze(output, dim=0)
        return output
    
################
# initial net
################
def init_net(net, init_type='normal', init_gain=0.02):
    if init_type != 'max' and init_type != 'none':
        print("Init Type:", init_type)
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        print("Init Type: Self-Normalizing Weights")
    return net

def init_weights(net, init_type='orthogonal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

################
# Regularization
################
def regularize_weights(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def define_reg(model):
    loss_reg = regularize_weights(model=model)
    return loss_reg

################
# Network Utils
################
def define_net():
    net = MaxNet(input_dim=60446, dropout_rate=0.25, init_max=True)
   
    return init_net(net, init_type = 'max')


def define_optimizer(model, lr, beta1=0.9 ,beta2=0.999, weight_decay=4e-4, optimizer_type='adam'):
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return optimizer


def define_scheduler(optimizer, lr_policy='linear', epoch_count=1, niter=0, niter_decay=50):
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    
    return scheduler

############
# Omic Model
############
class Aggregation_model(nn.Module):
    def __init__(self, inchannel, hidchannel, dropout = False, n_classes = 1):
        super(Aggregation_model, self).__init__()
        
        self.module_a = [nn.Linear(inchannel, hidchannel), nn.Linear(hidchannel, n_classes), nn.Tanh()]
        self.module_b = [nn.Linear(inchannel, hidchannel), nn.Linear(hidchannel, n_classes), nn.Sigmoid()]
        
        self.module_a = nn.Sequential(*self.module_a)
        self.module_b = nn.Sequential(*self.module_b)
        
    def forward(self, x):
        weight_a = self.module_a(x) 
        weight_b = self.module_b(x)
        weight = weight_a.mul(weight_b)
        weight = torch.transpose(weight, 1, 0)
        weight = F.softmax(weight, dim=1)
        result = torch.mm(weight, x)
        
        return result
    
class MaxNet(nn.Module):
    def __init__(self, input_dim=60446, dropout_rate=0.25, label_dim=2, init_max=True):
        super(MaxNet, self).__init__()
        #hidden = [64, 48, 32, 32]
        hidden = [2560, 256, 256, 256]
        
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        self.coattn = Attention(embed_dim=256, out_dim=256)
        self.Aggregation = Aggregation_model(inchannel=256, hidchannel=256)
        self.classifier = nn.Sequential(nn.Linear(256, label_dim))

        if init_max: init_max_weights(self)

    def forward(self, path_coattention=None, x=None, add_attention=0):
        x1 = self.encoder1(x)
        coattention = x1.view(10,256)
        
        if add_attention:
            features = self.coattn(coattention, path_coattention)
        else:
            features = self.Aggregation(coattention)
         
        out = self.classifier(features)
        
        return coattention, features, out

