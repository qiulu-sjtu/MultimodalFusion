import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,Dataset
import torch.backends.cudnn as cudnn
from torchvision import transforms,models
import torch.nn.functional as F 
 
class BasicBlock(nn.Module): 
    def __init__(self,in_num,out_num,cc=0):
        super(BasicBlock,self).__init__()
        self.in_num=in_num
        self.out_num=out_num
        self.cc=cc
        
        self.conv1=nn.Conv2d(self.in_num,self.out_num,kernel_size=3,padding=1)
        self.bn1=nn.InstanceNorm2d(self.out_num)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(self.out_num,self.out_num,kernel_size=3,padding=1)
        self.bn2=nn.InstanceNorm2d(self.out_num)
        self.relu2=nn.ReLU(inplace=True)
        
        if self.cc==1:
            self.conv3=nn.Conv2d(self.in_num,self.out_num,kernel_size=1)
    
    def forward(self,x):
        residual=x
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu1(out)
        out=self.conv2(out)
        out=self.bn2(out)
        
        if residual.size(1)!=out.size(1):
            residual=self.conv3(residual)
        
        out=out+residual
        out=self.relu2(out)
        
        return out

class Net(nn.Module):
    def __init__(self,block,f_list=[16,32,64,128,256],return_fea=False):
        super(Net,self).__init__()
        self.f_list=f_list
        
        self.down1=nn.Conv2d(self.f_list[0],self.f_list[0],kernel_size=2,stride=2)
        self.down2=nn.Conv2d(self.f_list[1],self.f_list[1],kernel_size=2,stride=2)
        self.down3=nn.Conv2d(self.f_list[2],self.f_list[2],kernel_size=2,stride=2)
        self.down4=nn.Conv2d(self.f_list[3],self.f_list[3],kernel_size=2,stride=2)
        
        self.layer1=self._make_layer_(block,3,self.f_list[0],1)
        self.layer2=self._make_layer_(block,self.f_list[0],self.f_list[1],1)
        self.layer3=self._make_layer_(block,self.f_list[1],self.f_list[2],1)
        self.layer4=self._make_layer_(block,self.f_list[2],self.f_list[3],2)
        self.layer5=self._make_layer_(block,self.f_list[3],self.f_list[4],2)
        
        self.AAP=nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Linear(self.f_list[4],128)
        self.relu1=nn.ReLU(inplace=True)
        self.drop=nn.Dropout(0.5)
        self.fc2=nn.Linear(128, 1)
        
        self.return_fea = return_fea
            
    def _check_input_dim_(self,x):
        if x.dim()!=4:
            raise ValueError('expect 4D input(got {}D input)'.format(x.dim()))
    
    def _make_layer_(self,block,in_num,out_num,depth):
        layers=[]
        for i in range(depth):
            if i==0:
                layers.append(block(in_num,out_num,cc=1))
            else:
                layers.append(block(out_num,out_num,cc=0))
        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        self._check_input_dim_(x)
        
        out1=self.layer1(x)
        down1=self.down1(out1)
        
        out2=self.layer2(down1)
        down2=self.down2(out2)
        
        out3=self.layer3(down2)
        down3=self.down3(out3)
        
        out4=self.layer4(down3)
        down4=self.down4(out4)
        
        out5=self.layer5(down4)
        
        out_aap=self.AAP(out5)
        out_aap=out_aap.view(-1,self.f_list[-1])
        
        out_fc1=self.fc1(out_aap)
        out_fc1=self.relu1(out_fc1)
        out_fc1_drop=self.drop(out_fc1)
        out_fc2=self.fc2(out_fc1_drop)
        
        if self.return_fea:
            return out_fc1_drop, torch.sigmoid(out_fc2)
        else:
            return torch.sigmoid(out_fc2)
    