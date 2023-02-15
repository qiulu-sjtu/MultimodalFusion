import os
import random
#import cv2
import math
import time
import csv
import numpy as np 
import scipy.io as scio
import PIL.Image as Image
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, roc_curve, auc

import argparse
parse = argparse.ArgumentParser('Train_Unimodal')
parse.add_argument('--gpu',type=str,default='0') 
parse.add_argument('--patient_csv',type=str)
parse.add_argument('--path_lib',type=str,default='./weights/features/')
parse.add_argument('--omic_lib',type=str,default='./DATA/OMIC/')
parse.add_argument('--output_file',type=str,default='./weights/Fusion/')
parse.add_argument('--model_type',type=str,default='last')
parse.add_argument('--cross',type=int)  #int, 0-4, select from 5-fold
parse.add_argument('--modality',type=str) #'path'/'omic
parse.add_argument('--max_epoch',type=int)
parse.add_argument('--step_size',type=int)
parse.add_argument('--batch_size',type=int)
parse.add_argument('--lr',type=float)
args = parse.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
 

import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms,models
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

from model.SNN_model import define_net, define_optimizer
from model.GCN import GraphNet
from evaluation import calculate_metrics

#### setting ###########
cross = args.cross
modality = args.modality

train_cross = [cross%5,(cross+1)%5,(cross+2)%5]
val_cross = [(cross+3)%5]

csv_file = args.patient_csv
path_lib = os.path.join(args.path_lib, 'cross'+str(cross))
omic_lib = args.omic_lib
output_file = os.path.join(args.output_file, 'cross'+str(cross), modality)
load_model_test = os.path.join(output_file, 'checkpoint_'+args.model_type+'.pth')

lr = args.lr
max_epoch = args.max_epoch
step_size = args.step_size

#-----------------------------------------------------------------------------------------------#
# loss
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss,self).__init__()
        
    def forward(self,y_pred, target):
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(y_pred, target)
        return loss

#-----------------------------------------------------------------------------------------------#
# data loader
class DataSet(Dataset):
    def __init__(self, cross_list, pathfile=path_lib, omicfile=omic_lib):
    
        csv_names = ['LUAD-', 'LUSC-']
        fold_names = ['TCGA-LUAD', 'TCGA-LUSC']
        
        # list patient & label
        self.paths = []
        self.omics = []
        self.targets = []
        
        for i in range(2):
            for j in cross_list:
                csv_path = os.path.join(csv_file, csv_names[i]+str(j)+'.csv')
                with open(csv_path) as f:
                    rows = csv.reader(f, delimiter = ',')
                    for row in rows:
                        uid = row[0]
                        
                        path_fold = os.path.join(pathfile, fold_names[i], uid)
                        omic_path = os.path.join(omicfile,fold_names[i],uid+'.mat')
                        
                        if not os.path.exists(path_fold):
                            print('File ERROR! :', path_fold)
                            continue
                        if not os.path.exists(omic_path):
                            print('File ERROR! :', omic_path)
                            continue
                        
                        imgs = os.listdir(path_fold)
                        for img in imgs:
                            path_path = os.path.join(path_fold, img)
                            if not os.path.exists(path_path):
                                print('File ERROR! :', path_path)
                            else:
                                self.paths.append(path_path)
                                self.omics.append(omic_path)
                                self.targets.append(i)
                            
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self,index):
        path_path = self.paths[index]
        omic_path = self.omics[index]
        
        # open path
        data_mat_path = scio.loadmat(path_path)
        path_features = torch.from_numpy(data_mat_path['feature']) 
        #roi_seq = features.unsqueeze(0)
        edge_index = torch.from_numpy(data_mat_path['edge']) 
            
        # open omic
        data_mat_omic = scio.loadmat(omic_path)
        omic_raw = torch.from_numpy(data_mat_omic['raw'])
            
        target = np.array([self.targets[index]])
        
        return path_features, edge_index, omic_raw, torch.from_numpy(target) 

#-----------------------------------------------------------------------------------------------#
# train
def train_step(data_set, mode, epoch, criterion, net, optim):
    if mode == 'test':
        net.eval()
    else:
        net.train()
        print(30*'-', '\n[epoch', epoch+1,'] Learning Rate:{}'.format(optim.param_groups[0]['lr']))
    
    # metrics
    pred_epoch, true_epoch = None, None
    running_loss = 0.
    consume_time = 0.
        
    data_len = data_set.__len__()
    data_shuffle = random.sample(range(data_len),data_len)
    step = 0
    for i in data_shuffle:
        now = time.time()
        path_features, edge_index, omic_raws, targets  = data_set.__getitem__(i)
        path_features, edge_index = path_features.cuda(), edge_index.cuda()
        omic_raws = omic_raws.to(torch.float32).cuda()
        targets = targets.cuda()
            
        data = Data(x=path_features, edge_index=edge_index)
        
        if mode != 'train':
            with torch.no_grad():
                if modality == 'path':
                    _, _, out = net(data = data)
                    loss = criterion(out, targets)
                    
                elif modality == 'omic':
                    _, _, out = net(x = omic_raws)
                    loss = criterion(out, targets)
                    
        else:
            optim.zero_grad()
            
            if modality == 'path':
                _, _, out = net(data = data)
                loss = criterion(out, targets)
                
            elif modality == 'omic':
                _, _, out = net(x = omic_raws)
                loss = criterion(out, targets)
                
            loss.backward()
            optim.step()
        
        # loss
        running_loss = (running_loss*step+loss.item())/(step+1)
        # acc auc
        out = nn.Softmax(dim=1)(out)
        pred_cpu = out.cpu().detach().numpy()
        true_cpu = targets.cpu().detach().numpy()
        pred_epoch = np.concatenate((pred_epoch, pred_cpu), axis=0) if step!=0 else pred_cpu.copy()
        true_epoch = np.concatenate((true_epoch, true_cpu), axis=0) if step!=0 else true_cpu.copy()
        
        running_acc, AUC, AP, f1score, sen, spe = calculate_metrics(pred_epoch, true_epoch)
        
        if mode != 'train':
            print("\r[Validation] Step {}/{}, loss:{:.5f}, acc:{:.4f}, AUC:{:.4f}, AP:{:.4f}, f1:{:.4f}, sen:{:.4f}, spe:{:.4f}".\
                  format(step, data_len, running_loss, running_acc, AUC, AP, f1score, sen, spe), end='')
        else:
            print('\r[Train] Step {}/{}, loss:{:.5f}, acc:{:.4f}, AUC:{:.4f}, AP:{:.4f}, f1:{:.4f}, sen:{:.4f}, spe:{:.4f}'.\
                  format(step, data_len, running_loss, running_acc, AUC, AP, f1score, sen, spe), end='')
        consume_time += time.time() - now
        step += 1
   
    print(' ')

    return running_loss, running_acc, AUC, AP, f1score, sen, spe, consume_time/step

#-----------------------------------------------------------------------------------------------#
# train main
def train(max_epoch=max_epoch, ModelSet = '', save_csv = True, save_best_model = True, save_last_model = True, isload = False, load_model_name=''):
    
    print('-----------------start train unimodal-----------------------') 
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    random.seed(0)
    
    if modality == 'omic':
        model = define_net()
        model.cuda()
        optimizer = define_optimizer(model=model, lr=lr)
        #scheduler = define_scheduler(optimizer)
        scheduler = lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=0.5)
    elif modality == 'path':
        feature_length = 128
        model = GraphNet(feature_length=feature_length, nhid=256)
        model.cuda() 
        optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=0.5)
        
    criterion = CrossEntropyLoss().cuda()
    
    if isload and os.path.exists(load_model_name):
        pretrained=torch.load(load_model_name)
        model.load_state_dict(pretrained['state_dict'])
        print('model load successfully...')
        
    start_epoch = 0
        
    lowest_loss = float('Inf')  # 寻找最小loss
        
    train_set = DataSet(train_cross)
    val_set = DataSet(val_cross)
    
    for epoch in range(start_epoch, max_epoch):
        # train
        running_loss, running_acc, running_auc, ap, f1, sen, spe, consume_time = \
            train_step(train_set, 'train', epoch, criterion, model, optimizer)
        scheduler.step()
        
        # validation
        val_loss, val_acc, val_auc, val_ap, val_f1, val_sen, val_spe, consume_time = \
            train_step(val_set, 'test', epoch, criterion, model, optimizer)
        
        # save best model
        print('')
        if val_loss < lowest_loss:
            print('Val_loss is lower than previous {:.5f}. The new best model saved. '.format(lowest_loss))
            lowest_loss = val_loss
            #obj = {'state_dict': model.state_dict()}
                
            #torch.save(obj, os.path.join(ModelSet,'checkpoint_best.pth'))
            #print('Current loss < initial loss, best model save......')
            print('Current loss < initial loss......')
        else:
            print('Val_loss_all is not improved from {:.5f}. '.format(lowest_loss))
        
    # save model
    obj = {'state_dict': model.state_dict()}
                
    torch.save(obj, os.path.join(ModelSet,'checkpoint_last.pth'))
    print('Last model saved. ')

    print('program ends')    

    
if __name__ == "__main__":
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    print('cross: ', cross)
    print('modality: ', modality)
    
    train(ModelSet = output_file)
    