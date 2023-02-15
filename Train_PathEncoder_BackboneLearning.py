import os
import sys
import numpy as np
import PIL.Image as Image
import random
from glob import glob
import csv
import time
import math

import torch 
import torch.nn as nn 
import torch.optim as opt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,Dataset 
import torch.backends.cudnn as cudnn
from torchvision import transforms,models
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc
#from torch.utils import data

import argparse
parse = argparse.ArgumentParser('Encoder')
parse.add_argument('--patient_csv',type=str)
parse.add_argument('--data_lib',type=str, default = './DATA/PATH-patch/')
parse.add_argument('--output_file',type=str, default = './weights/PathIn/')
parse.add_argument('--gpu',type=str,default='0')
parse.add_argument('--cross',type=int)  #int, 0-4, select from 5-fold
parse.add_argument('--max_epoch',type=int)
parse.add_argument('--batch_size',type=int)
parse.add_argument('--lr',type=float)
parse.add_argument('--step_size',type=int)
parse.add_argument('--patience',type=int)
args = parse.parse_args()

from model.networks import Net, BasicBlock


# setting
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

cross = args.cross
train_cross = [cross,(cross+1)%5,(cross+2)%5]
val_cross = [(cross+3)%5]

batch_size = args.batch_size
lr = args.lr
step_size = args.step_size
max_epoch = args.max_epoch

csv_file = args.patient_csv
patch_lib = args.data_lib
output_file = os.path.join(args.output_file, 'cross'+str(cross))

isload = False
load_model_name = os.path.join(output_file, 'checkpoint_last.pth')


############### dataset ##############################################################
class DataSet(Dataset):
    def __init__(self, libraryfile, cross_list, transform=None):
        self.root = libraryfile
        
        csv_names = ['LUAD-', 'LUSC-']
        fold_names = ['TCGA-LUAD', 'TCGA-LUSC']
        
        # list patient & label
        self.grids = []
        self.targets = []
        
        for i in range(2):
            for j in cross_list:
                csv_path = os.path.join(csv_file, csv_names[i]+str(j)+'.csv')
                with open(csv_path) as f:
                    rows = csv.reader(f, delimiter = ',')
                    for row in rows:
                        uid = row[0]
                        uid_path = os.path.join(self.root, fold_names[i], uid)
                        if not os.path.exists(uid_path):
                            print('File ERROR! :', uid_path)
                        else:
                            imgs = os.listdir(uid_path)
                            for img in imgs:
                                self.grids.append(os.path.join(uid_path, img))
                                self.targets.append(i)             
        
        self.transform = transform
        
    def __len__(self):
        return len(self.grids)
        
    def __getitem__(self,index):
        patchs_path = self.grids[index]
        try:
            image = Image.open(patchs_path)  
        
        except FileNotFoundError:
            print('Unable to find {} file'.format(patchs_path))
            return None
        
        if self.transform is not None:
            image = self.transform(image)
        
        target = np.array([self.targets[index]])
        return image, target, index
    
        
############### loss ##############################################################
class OneCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(OneCrossEntropyLoss,self).__init__()
    
    def forward(self,y_pred,y_true):
        p=y_pred.squeeze()
        y_true=y_true.squeeze()
        cc=y_true*torch.log(p+1e-8)+(1.0-y_true)*torch.log(1.0-p+1e-8)
        cc_loss=-1.0*torch.mean(cc)
        return cc_loss

class ClassifyLoss(nn.Module):
    def __init__(self, num_classes):
        super(ClassifyLoss,self).__init__()
        self.num_classes = num_classes 
    def forward(self,y_pred, target):
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(y_pred, target)
            
        return loss
    
    
############### train ##############################################################
def train_step(net, data_loader, epoch, criterion, optimizer, mode='train'):
    
    if mode != 'train':
        net.eval()
    else:
        net.train()
        print(30*'-', '\n[epoch', epoch+1,'] Learning Rate:{}'.format(optimizer.param_groups[0]['lr']))
    
    y_true_list, y_pred_list = [],[]
    running_loss = 0.
    consume_time = 0.
    
    for batch_idx, (inputs, targets, indexes) in enumerate(data_loader):
        now = time.time()
        inputs, targets = inputs.cuda(), targets.cuda()
        targets = targets.squeeze(-1)
        
        if mode != 'train':
            with torch.no_grad():
                out = net(inputs)
                loss = criterion(out, targets)
        else:
            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            
        # loss
        running_loss = (running_loss*batch_idx+loss.item())/(batch_idx+1)
        # acc auc
        #out = nn.Softmax(dim=1)(out)
        y_true = list(targets.cpu().detach().numpy().squeeze())
        y_true_list.extend(y_true)
        y_pred = list(out.cpu().detach().numpy().squeeze())
        y_pred_list.extend(y_pred)
        y_pred_list_1 = [1 if x >= 0.5 else 0 for x in y_pred_list]
        if sum(y_true_list)!=0 and sum(y_true_list)!=len(y_true_list):
            fpr, tpr, thresholds = roc_curve(np.array(y_true_list), np.array(y_pred_list))
            AUC = auc(fpr, tpr)
        else:
            AUC = 0 
        corrects = np.sum(np.array(y_pred_list_1)==np.array(y_true_list))
        running_acc = corrects/len(y_true_list)
        
        if mode != 'train':
            print("\r[Validation] Step {}/{}, loss:{:.5f}, acc:{:.4f}, AUC:{:.3f}, time_cost:{:.2f}s".format(batch_idx+1, len(data_loader), running_loss, running_acc, AUC, consume_time), end='')
        else:
            print('\r[Train] Step {}/{}, loss:{:.5f}, acc:{:.4f}, AUC:{:.3f}, time_cost:{:.2f}s'.format(batch_idx+1, len(data_loader), running_loss, running_acc, AUC, consume_time), end='')
        consume_time += time.time() - now
    
    cm = confusion_matrix(y_pred=y_pred_list_1, y_true=y_true_list)
    print(' ')
    print(cm)
    return running_loss, running_acc, AUC, consume_time/(batch_idx+1)
    
    
def main(max_epoch):
    print('start "main" program')
    #cnn
    model = Net(BasicBlock)
    model.cuda()
    cudnn.benchmark = True
    
    optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.5)

    if isload and os.path.exists(load_model_name):
        pretrained=torch.load(load_model_name)
        model.load_state_dict(pretrained['state_dict'])
        print('model load successfully: '+load_model_name)
    

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    train_trans = transforms.Compose([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.2),\
                                      transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.5),\
                                      transforms.ToTensor(), normalize])
    valid_trans= transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = DataSet(patch_lib, train_cross, train_trans)
    train_loader = torch.utils.data.DataLoader(train_dset,batch_size=batch_size, shuffle=True,\
                                               num_workers=4, pin_memory=False)
    val_dset = DataSet(patch_lib, val_cross, valid_trans)
    val_loader = torch.utils.data.DataLoader(val_dset,batch_size=batch_size, shuffle=True,\
                                                 num_workers=4, pin_memory=False)
    
    criterion = OneCrossEntropyLoss().cuda()
    
    #open output file
    if not os.path.exists(output_file):
        os.makedirs(output_file)
        
    #train
    print('-'*30)
    print('start training...')
    print('-'*30)
    lowest_loss = float('Inf')  # 寻找最小loss
    patience = 0
    start_epoch = 0
    
    for epoch in range(start_epoch, max_epoch):
        # train dataset
        
        running_loss, running_acc, AUC, consume_time = train_step(model, train_loader, epoch, criterion, optimizer, mode='train')
        # update scheduler
        scheduler.step()
        
        # validation dataset
        val_loss, val_acc, val_AUC, mconsume_time = train_step(model, val_loader, epoch, criterion, optimizer, mode='test')
        
        # save best model
        patience += 1
        
        if val_loss < lowest_loss:
            print('Val_loss is lower than previous {:.5f}. The new best model saved. '.format(lowest_loss))
            lowest_loss = val_loss
            obj = {'state_dict': model.state_dict()}
                
            torch.save(obj, os.path.join(output_file,'checkpoint_best.pth'))
            print('Current loss < initial loss, best model save......')
            patience = 0
        else:
            print('Val_loss_all is not improved from {:.5f}. '.format(lowest_loss))
        
        # save model
        obj = {'state_dict': model.state_dict()}
                
        torch.save(obj, os.path.join(output_file,'checkpoint_last'+str(epoch)+'.pth'))
        print('Last model saved. ')
        
        if patience >=args.patience:
            break

    print('program ends')

if __name__ == '__main__':
    main(max_epoch)
      
        

       