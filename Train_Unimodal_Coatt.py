import os
#import cv2
import math
import time
import csv
import random
import numpy as np
import scipy.io as scio 
import PIL.Image as Image
#from sklearn.metrics import confusion_matrix, roc_curve, average_precision_score, auc, f1_score

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

cross = args.cross
modality = args.modality

train_cross = [cross%5,(cross+1)%5,(cross+2)%5]
val_cross = [(cross+3)%5]

csv_file = args.patient_csv
path_lib = os.path.join(args.path_lib, 'cross'+str(cross))
omic_lib = args.omic_lib
output_file = os.path.join(args.output_file, 'cross'+str(cross), modality+'_coatt')

path_coatt_feature = os.path.join(args.output_file, 'cross'+str(cross), 'coatt_feature/path')
omic_coatt_feature = os.path.join(args.output_file, 'cross'+str(cross), 'coatt_feature/omic')

load_coatt_path_name = os.path.join(args.output_file, 'cross'+str(cross), 'path/checkpoint_last.pth')
load_coatt_omic_name = os.path.join(args.output_file, 'cross'+str(cross), 'omic/checkpoint_last.pth')
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
    def __init__(self, cross_list, pathfile=path_lib, omicfile=omic_lib, pathcoattfile=path_coatt_feature, omiccoattfile=omic_coatt_feature):
        # list patient & label
        self.paths = []
        self.path_paths = []
        self.omics = []
        self.omic_paths = []
        self.targets = []
        
        csv_names = ['LUAD-', 'LUSC-']
        fold_names = ['TCGA-LUAD', 'TCGA-LUSC']
        
        for i in range(2):
            for j in cross_list:
                csv_path = os.path.join(csv_file, csv_names[i]+str(j)+'.csv')
                with open(csv_path) as f:
                    rows = csv.reader(f, delimiter = ',')
                    for row in rows:
                        uid = row[0]
                        label = i
        
                        path_fold = os.path.join(pathfile, fold_names[i], uid)
                        omic_path = os.path.join(omicfile, fold_names[i], uid+'.mat')
                        omic_fea_path = os.path.join(omic_coatt_feature, uid+'.mat')
                    
                        if not os.path.exists(omic_path):
                            print('File ERROR! :', omic_path)
                            continue
                    
                        if not os.path.exists(path_fold):
                            print('File ERROR! :', path_fold)
                            continue
                    
                        imgs = os.listdir(path_fold)
                        for img in imgs:
                            path_path = os.path.join(path_fold, img)
                            path_fea_path = os.path.join(path_coatt_feature, uid, img)
                        
                            if not os.path.exists(path_path):
                                print('File ERROR! :', path_path)
                            else:
                                self.paths.append(path_path)
                                self.path_paths.append(path_fea_path)
                                self.omics.append(omic_path)
                                self.omic_paths.append(omic_fea_path)
                                self.targets.append(int(label))
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self,index):
        path_path = self.paths[index]
        path_fea_path = self.path_paths[index]
        omic_path = self.omics[index]
        omic_fea_path = self.omic_paths[index]
        
        # open path
        data_mat_path = scio.loadmat(path_path)
        path_features = torch.from_numpy(data_mat_path['feature']) 
        #roi_seq = features.unsqueeze(0)
        edge_index = torch.from_numpy(data_mat_path['edge']) 
        path_fea = torch.from_numpy(scio.loadmat(path_fea_path)['path_feature'])
            
        # open omic
        data_mat_omic = scio.loadmat(omic_path)
        omic_raw = torch.from_numpy(data_mat_omic['raw'])
        omic_fea = torch.from_numpy(scio.loadmat(omic_fea_path)['omic_feature'])
        
        target = np.array([self.targets[index]])
        
        return path_features, edge_index, omic_raw, torch.from_numpy(target), path_fea, omic_fea

#-----------------------------------------------------------------------------------------------#
# get coatt feature
def inference(model_coattn, modal):
    print('-----------------inference coattention feature-----------------------')
    model_coattn.eval()
    
    # metrics
    pred_epoch, true_epoch = None, None
    
    types = ['TCGA-LUAD', 'TCGA-LUSC']
     
    for i in range(2):
        uids = os.listdir(os.path.join(path_lib, types[i]))
        path_path = os.path.join(path_lib, types[i])
        omic_path = os.path.join(omic_lib, types[i])
        
        for uid in uids:
            if modal == 'omic':
                mat_path = os.path.join(omic_path, uid+'.mat')
                if not os.path.exists(mat_path):
                    continue
                data_mat_omic = scio.loadmat(mat_path)
                omic_raw = torch.from_numpy(data_mat_omic['raw'])
                omic_raw = omic_raw.to(torch.float32).cuda()
                target = np.array([i])
            
                att_features, _, out = model_coattn(x = omic_raw)
                att_features = att_features.cpu().detach().numpy()
                if not os.path.exists(omic_coatt_feature):
                    os.makedirs(omic_coatt_feature)
                scio.savemat(os.path.join(omic_coatt_feature,uid+'.mat'), {'omic_feature':att_features})
                
                out = nn.Softmax(dim=1)(out)
                pred_cpu = out.cpu().detach().numpy()
                true_cpu = target
                pred_epoch = np.concatenate((pred_epoch, pred_cpu), axis=0) if (pred_epoch is not None) else pred_cpu.copy()
                true_epoch = np.concatenate((true_epoch, true_cpu), axis=0) if (true_epoch is not None) else true_cpu.copy()
        
                acc, AUC, AP, f1score, sen, spe = calculate_metrics(pred_epoch, true_epoch)
                print("\r[omic] acc:{:.4f}, AUC:{:.4f}, AP:{:.4f}, f1:{:.4f}".format(acc, AUC, AP, f1score), end='')
            
            if modal == 'path':
                if not os.path.exists(os.path.join(path_path, uid)):
                    continue
                imgs = os.listdir(os.path.join(path_path, uid))
                for img in imgs:
                    mat_path = os.path.join(path_path, uid, img)
                    data_mat_path = scio.loadmat(mat_path)
                    path_features = torch.from_numpy(data_mat_path['feature']).cuda()
                    edge_index = torch.from_numpy(data_mat_path['edge']) .cuda()
                    data = Data(x=path_features, edge_index=edge_index)
                    target = np.array([i])
                
                    att_features, _, out = model_coattn(data=data)
                    att_features = att_features.cpu().detach().numpy()
                    if not os.path.exists(os.path.join(path_coatt_feature, uid)):
                        os.makedirs(os.path.join(path_coatt_feature, uid))
                    scio.savemat(os.path.join(path_coatt_feature, uid, img), {'path_feature':att_features})
                    
                    out = nn.Softmax(dim=1)(out)
                    pred_cpu = out.cpu().detach().numpy()
                    true_cpu = target
                    pred_epoch = np.concatenate((pred_epoch, pred_cpu), axis=0) if (pred_epoch is not None) else pred_cpu.copy()
                    true_epoch = np.concatenate((true_epoch, true_cpu), axis=0) if (true_epoch is not None) else true_cpu.copy()
        
                    acc, AUC, AP, f1score, sen, spe = calculate_metrics(pred_epoch, true_epoch)
                    print("\r[path] acc:{:.4f}, AUC:{:.4f}, AP:{:.4f}, f1:{:.4f}".format(acc, AUC, AP, f1score), end='')

#-----------------------------------------------------------------------------------------------#
# train
def train_step(data_set, mode, epoch, criterion, net, optim):
    if mode == 'test':
        net.eval()
    else:
        net.train()
        print(30*'-', '\n[epoch', epoch+1,'] Learning Rate:{}'.format(optim.param_groups[0]['lr']))
    
    running_loss = 0.
    consume_time = 0.
        
    data_len = data_set.__len__()
    data_shuffle = random.sample(range(data_len),data_len)
    step = 0
    for i in data_shuffle:
        now = time.time()
        path_features, edge_index, omic_raws, targets, path_fea, omic_fea = data_set.__getitem__(i)
        path_features, edge_index = path_features.cuda(), edge_index.cuda()
        omic_raws = omic_raws.to(torch.float32).cuda()
        targets = targets.cuda()
        path_fea, omic_fea = path_fea.cuda(), omic_fea.cuda()
            
        data = Data(x=path_features, edge_index=edge_index)
        
        if mode != 'train':
            with torch.no_grad():
                if modality == 'path':
                    _, _, out = net(data, omic_fea, 1)
                    loss = criterion(out, targets)
                    
                elif modality == 'omic':
                    _, _, out = net(path_fea, omic_raws, 1)
                    loss = criterion(out, targets)
                    
        else:
            optim.zero_grad()
            
            if modality == 'path':
                _, _, out = net(data, omic_fea, 1)
                loss = criterion(out, targets)
                
            elif modality == 'omic':
                _, _, out = net(path_fea, omic_raws, 1)
                loss = criterion(out, targets)
                
            loss.backward()
            optim.step()
        
        # loss
        running_loss = (running_loss*step+loss.item())/(step+1)
        
        if mode != 'train':
            print("\r[Validation] Step {}/{}, loss:{:.5f}".\
                  format(step, data_len, running_loss), end='')
        else:
            print('\r[Train] Step {}/{}, loss:{:.5f}'.\
                  format(step, data_len, running_loss), end='')
        consume_time += time.time() - now
        step += 1
   
    print(' ')

    return running_loss, consume_time/step


#-----------------------------------------------------------------------------------------------#
# train main   
def pretrain_unimodal_coattention(max_epoch=max_epoch, ModelSet = '', save_csv = True, save_best_model = True, save_last_model = True, \
                                  isload = False, load_model_name=''):
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    random.seed(0)
    
    # get coatt feature
    if not os.path.exists(path_coatt_feature):
        model = GraphNet(feature_length=128, nhid=256)
        model.cuda()
        pretrained=torch.load(load_coatt_path_name)
        model.load_state_dict(pretrained['state_dict'])
        print('coattn path model load successfully: ',load_coatt_path_name)
        inference(model, 'path')
        del model
    if not os.path.exists(omic_coatt_feature):
        model = define_net()
        model.cuda()
        pretrained=torch.load(load_coatt_omic_name)
        model.load_state_dict(pretrained['state_dict'])
        print('coattn omic model load successfully: ',load_coatt_omic_name)
        inference(model, 'omic')
        del model
            
    # define net
    if modality == 'omic':
        model = define_net()
        model.cuda()
        optimizer = define_optimizer(model=model, lr=lr)
        #scheduler = define_scheduler(optimizer)
        scheduler = lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=0.5)
    elif modality == 'path':
        model = GraphNet(feature_length=128, nhid=256)
        model.cuda() 
        optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=0.5)
        
    criterion = CrossEntropyLoss().cuda()
    
    start_epoch = 0  
    lowest_loss = float('Inf')  # 寻找最小loss
        
    train_set = DataSet(train_cross)
    val_set = DataSet(val_cross)
    
    for epoch in range(start_epoch, max_epoch):
        # train dataset
        running_loss, consume_time = \
            train_step(train_set, 'train', epoch, criterion, model, optimizer)
        
        scheduler.step()
        
        # validation dataset
        val_loss, consume_time = \
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
  
    
if __name__ == "__main__":
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    print('-----------------start pretrain unimodal-----------------------') 
    print('cross: ', cross)
    print('pretrain modality: ', modality)
    
    pretrain_unimodal_coattention(ModelSet = output_file)
    