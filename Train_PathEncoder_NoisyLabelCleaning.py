import os
import sys
import argparse
import numpy as np
import PIL.Image as Image
import random
from glob import glob
import csv
import time 
import pickle 

import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,Dataset 
import torch.backends.cudnn as cudnn
from torchvision import transforms,models
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
parse.add_argument('--lamda',type=float)
parse.add_argument('--step_size',type=int)
parse.add_argument('--patience',type=int)
args = parse.parse_args()

from model.networks import Net, BasicBlock

# setting
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

cross = args.cross
train_cross = [cross,(cross+1)%5,(cross+2)%5]
val_cross = [(cross+3)%5]

lamda = args.lamda
batch_size = args.batch_size
lr = args.lr
step_size = args.step_size
max_epoch = args.max_epoch

csv_file = args.patient_csv
patch_lib = args.data_lib
output_file = os.path.join(args.output_file,'cross'+str(cross))

isload = False
load_model_name = os.path.join(output_file, 'checkpoint_last.pth')
load_pretrain_name = os.path.join(output_file, 'checkpoint_best.pth')
pretrain_result_train = os.path.join(output_file, 'pretrain_train.pkl')
pretrain_result_val = os.path.join(output_file, 'pretrain_val.pkl')

########## get pretrain model prediction #######################################
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)   
        
def PretrainModel_Predict(model, result_file, root='', cross_list=[], transform=None):
    csv_names = ['LUAD-', 'LUSC-']
    fold_names = ['TCGA-LUAD', 'TCGA-LUSC']
   
    grids = []
    targets = []
    
    for i in range(2):
        for j in cross_list:
            csv_path = os.path.join(csv_file, csv_names[i]+str(j)+'.csv')
            with open(csv_path) as f:
                rows = csv.reader(f, delimiter = ',')
                for row in rows:
                    uid = row[0]
                    uid_path = os.path.join(root, fold_names[i], uid)
                    imgs = os.listdir(uid_path)
                    for img in imgs:
                        grids.append(os.path.join(uid_path, img))
                        targets.append(i) 
    
    index = 0
    patch_batch, target_batch = None, None
    pred_epoch, true_epoch = None, None
    
    #for i in range(len(grids)):
    for i in range(20):
        patch_path = grids[i]
        image = Image.open(patch_path)
        #print(index)
        image = transform(image).unsqueeze(0)
        target = torch.Tensor([targets[i]]).unsqueeze(0)
        
        patch_batch = torch.cat((patch_batch, image), dim=0) if index!=0 else image.clone()
        target_batch = torch.cat((target_batch, target), dim=0) if index!=0 else target.clone()
        index+=1
        
        if index == batch_size:
            patch_batch = patch_batch.cuda()
            target_batch = target_batch.cuda()
            with torch.no_grad():
                probs = model(patch_batch)
            
            pred_cpu = probs.cpu().detach().numpy()
            pred_epoch = np.concatenate((pred_epoch, pred_cpu), axis=0) if pred_epoch is not None else pred_cpu.copy()
            true_cpu = target_batch.cpu().detach().numpy()
            true_epoch = np.concatenate((true_epoch, true_cpu), axis=0) if true_epoch is not None else true_cpu.copy()
            
            y_true_list = list(true_epoch.squeeze())
            y_pred_list = list(pred_epoch.squeeze())
            y_pred_list_1 = [1 if x >= 0.5 else 0 for x in y_pred_list]
            corrects = np.sum(np.array(y_pred_list_1)==np.array(y_true_list))
            running_acc = corrects/len(y_true_list)
            
            print('\rStep {}/{}, acc:{:.4f}'.format(true_epoch.shape[0], len(grids), running_acc), end='')
            index = 0
        
    if index !=0:
        patch_batch = patch_batch.cuda()
        target_batch = target_batch.cuda()
        with torch.no_grad():
            probs = model(patch_batch)
        
        pred_cpu = probs.cpu().detach().numpy()
        pred_epoch = np.concatenate((pred_epoch, pred_cpu), axis=0) if pred_epoch is not None else pred_cpu.copy()
        true_cpu = target_batch.cpu().detach().numpy()
        true_epoch = np.concatenate((true_epoch, true_cpu), axis=0) if true_epoch is not None else true_cpu.copy()
            
        y_true_list = list(true_epoch.squeeze())
        y_pred_list = list(pred_epoch.squeeze())
        y_pred_list_1 = [1 if x >= 0.5 else 0 for x in y_pred_list]
        corrects = np.sum(np.array(y_pred_list_1)==np.array(y_true_list))
        running_acc = corrects/len(y_true_list)
        
        print('\rStep {}/{}, acc:{:.4f}'.format(true_epoch.shape[0], len(grids), running_acc), end='')
            
    print(' ')        
    dict_result = {'girds':grids, 'hard':true_epoch, 'soft':pred_epoch}
    save_obj(dict_result, result_file)
            

########## dataset ############################################################
def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
class DataSet(Dataset):
    def __init__(self, libraryfile, transform=None):
        self.root = libraryfile
        
        obj = load_obj(self.root)
        self.grids = obj['girds']
        self.softs = obj['soft']
        self.hards = obj['hard']
        
        self.transform = transform
        #print(self.hards[0,:].shape)
        #print(self.softs[0,:].shape)
    
    def ChangeSoft(self, index, pro):
        self.softs[index, :] = pro

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
        
        target = np.array(int(self.hards[index, :]))
        soft = self.softs[index, :]
        INDEX = np.array([index])
        
        return image, target, soft, INDEX

############### loss ##############################################################    
class OneCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(OneCrossEntropyLoss,self).__init__()
    
    def forward(self, y_pred, y_true, alpha = 0.5):
        p=y_pred.squeeze()
        y_true=y_true.squeeze()
        cc=2*((1-alpha)*y_true*torch.log(p+1e-8) + alpha*(1.0-y_true)*torch.log(1.0-p+1e-8))
        cc_loss=-1.0*torch.mean(cc)
        return cc_loss

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss,self).__init__()
    
    def forward(self,y_pred,y_true):
        p=y_pred.squeeze()
        y_true=y_true.squeeze()
        c1 = torch.mean(p*torch.log((p+1e-8)/(y_true+1e-8)))
        
        p=1-y_pred.squeeze()
        y_true=1-y_true.squeeze()
        c2 = torch.mean(p*torch.log((p+1e-8)/(y_true+1e-8)))
        
        cc_loss=c1+c2
        return cc_loss
'''
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss,self).__init__()
        
    def forward(self,y_pred, target):
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(y_pred, target)
            
        return loss

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss,self).__init__()
    
    def forward(self,y_pred,y_true_soft,num_class=3):
        y_pred = nn.Softmax(dim=1)(y_pred)
        loss = None
        for i in range(y_pred.shape[-1]):
            p = y_pred[:,i]
            t = y_true_soft[:,i]
            c = torch.mean(p*torch.log((p+1e-8)/(t+1e-8)))
            loss = loss + c if (loss is not None) else c
        
        return loss
'''

############### train ##########################################################
def train_step(net, data_dset, epoch, criterion_h, criterion_s, optimizer, mode='train'): 
    data_loader = torch.utils.data.DataLoader(data_dset, batch_size=batch_size, shuffle=True,\
                        num_workers=4, pin_memory=False) 
        
    if mode != 'train':
        net.eval()
    else:
        net.train()
        print(30*'-', '\n[epoch', epoch+1,'] Learning Rate:{}'.format(optimizer.param_groups[0]['lr']))
    
    y_true_list, y_pred_list = [],[]
    running_loss = 0.
    consume_time = 0.
    
    for batch_idx, (imgs, labels_h, labels_s, _) in enumerate(data_loader):
        now = time.time()
        imgs, labels_h, labels_s = imgs.cuda(), labels_h.cuda(), labels_s.cuda()
        labels_h, labels_s = labels_h.squeeze(-1), labels_s.squeeze(-1)
        
        if mode != 'train':
            with torch.no_grad():
                out = net(imgs)
                loss_h = criterion_h(out, labels_h)
                loss_s = criterion_s(out, labels_s)
                loss = lamda*loss_h + (1-lamda)*loss_s
        else:
            optimizer.zero_grad()
            out = net(imgs)
            loss_h = criterion_h(out, labels_h)
            loss_s = criterion_s(out, labels_s)
            loss = lamda*loss_h + (1-lamda)*loss_s
            loss.backward()
            optimizer.step()
        
        # loss
        running_loss = (running_loss*batch_idx+loss.item())/(batch_idx+1)
        # acc auc
        y_true = list(labels_h.cpu().detach().numpy().squeeze())
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
            

def change_labels(net, data_dset):
    
    data_loader = torch.utils.data.DataLoader(data_dset, batch_size=batch_size, shuffle=False,\
                        num_workers=4, pin_memory=False) 
    net.eval()
    for i, data in enumerate(data_loader, 0):
        imgs, _, _, INDEX = data
        imgs = imgs.cuda()
        
        with torch.no_grad():
            out = net(imgs)
        y_pred = out.cpu().detach().numpy()
        for j in range(y_pred.shape[0]):
            pro = y_pred[j]
            index = np.array(INDEX[j]).squeeze()
            data_dset.ChangeSoft(int(index), pro)
        print("\r[infering] Step {}/{}".format(i+1, len(data_loader)), end='')
    print(' ')
    
    return data_dset
             

def main(max_epoch):
    #open output file
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    
    
    # normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    train_trans = transforms.Compose([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.2),\
                                      transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.5),\
                                      transforms.ToTensor(), normalize])
    valid_trans= transforms.Compose([transforms.ToTensor(), normalize])
    
    
    print('start "main" program')
    #cnn
    model = Net(BasicBlock)
    model.cuda()
    cudnn.benchmark = True
    
    #get pretrain model prediction
    pretrained=torch.load(load_pretrain_name)
    model.load_state_dict(pretrained['state_dict'])
    print('pretrained model load successfully: '+load_pretrain_name)
    if not os.path.exists(pretrain_result_train):
        PretrainModel_Predict(model, pretrain_result_train, patch_lib, train_cross, valid_trans)
    if not os.path.exists(pretrain_result_val):
        PretrainModel_Predict(model, pretrain_result_val, patch_lib, val_cross, valid_trans)
    
    #loss
    criterion_h = OneCrossEntropyLoss().cuda()
    criterion_s = KLDivergenceLoss().cuda()
    optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    if isload and os.path.exists(load_model_name):
        pretrained=torch.load(load_model_name)
        model.load_state_dict(pretrained['state_dict'])
        print('model load successfully: '+load_model_name)
    
    
    #load data
    train_dset = DataSet(pretrain_result_train, train_trans)
    val_dset = DataSet(pretrain_result_val, valid_trans)
    
    
    # train
    print('-'*30)
    print('start training...')
    print('-'*30)
    lowest_loss = float('Inf')  # ????????????loss
    patience = 0
    start_epoch = 0

    for epoch in range(start_epoch, max_epoch):
        
        # train dataset
        running_loss, AUC, running_acc, consume_time = \
            train_step(model, train_dset, epoch, criterion_h, criterion_s, optimizer, mode='train')
        # update scheduler
        scheduler.step()
        
        # validation dataset
        val_loss, val_AUC, val_acc, consume_time = \
            train_step(model, val_dset, epoch, criterion_h, criterion_s, optimizer, mode='test')
        
        # change softlabels
        if (epoch+1) % step_size == 0:
            print('start change train dataset...')
            train_dset = change_labels(model, train_dset)
            print('start change validation dataset...')
            val_dset = change_labels(model, val_dset)   
        
        patience += 1
        
        # save best model
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
                
        torch.save(obj, os.path.join(output_file,'checkpoint_last.pth'))
        print('Last model saved. ')
        
        if patience >=args.patience:
            break

    print('program ends')

if __name__ == "__main__":
    main(max_epoch)
