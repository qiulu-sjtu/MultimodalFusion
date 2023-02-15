# Torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor 
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.nn import Sequential as Seq
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GraphConv, GatedGraphConv, GATConv
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.transforms.normalize_features import NormalizeFeatures
#import os


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
    
############
# Graph Model
############
class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        data.x = data.x.type(torch.cuda.FloatTensor)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]#.type(torch.cuda.FloatTensor)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
  
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
  
    
class GraphNet(torch.nn.Module):
    def __init__(self, feature_length=128, nhid=512, grph_dim=512, nonlinearity=torch.tanh, 
        dropout_rate=0.25, GNN='GCN', use_edges=0, pooling_ratio=0.20, act=None, n_classes=2):
        super(GraphNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.use_edges = use_edges
        self.act = act
        self.conv1 = SAGEConv(feature_length, 256)
        self.conv2 = SAGEConv(256, nhid)
        self.conv3 = SAGEConv(nhid, nhid)
        self.conv4 = SAGEConv(nhid, nhid)
        self.conv5 = SAGEConv(nhid, nhid)
        
        self.coattn = Attention(embed_dim=nhid, out_dim=nhid)
        self.Aggregation = Aggregation_model(inchannel=nhid, hidchannel=nhid)
        self.classifier = torch.nn.Linear(nhid, n_classes)
        
    def forward(self, data=None, omic_features=None, add_attention=0):
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, None, data.batch
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        x4 = F.relu(self.conv4(x3, edge_index, edge_attr))
        x5 = F.relu(self.conv5(x4, edge_index, edge_attr))
        
        if add_attention:
            # x5 (n,256) omic_features (1,256)
            coattn = self.coattn(x5, omic_features) 
            #print(coattn.shape)
            out = self.Aggregation(coattn)
        else:
            out = self.Aggregation(x5)
        
        logits  = self.classifier(out)
        # out (1,256)
        #return out, out, nn.LogSoftmax(dim=1)(logits)
        return out, out, logits
