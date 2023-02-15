import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,Dataset
import torch.backends.cudnn as cudnn
from torchvision import transforms,models
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
import PIL.Image as Image
import random
import cv2
import csv
import math
from glob import glob 
import scipy.io as scio   

import argparse
parse = argparse.ArgumentParser('Encoder')
parse.add_argument('--gpu',type=str,default='0')
parse.add_argument('--cross',type=int)  #int, 0-4, select from 5-fold
parse.add_argument('--data_lib',type=str,default='./DATA/PATH-slide')
parse.add_argument('--output_file',type=str,default='./weights/features/')
parse.add_argument('--load_model_file',type=str,default='./weights/PathIn/')
parse.add_argument('--batch_size',type=int,default=16)
args = parse.parse_args()

from model.networks import Net, BasicBlock

# setting
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

Image.MAX_IMAGE_PIXELS = None

patch_size = 512
cross = args.cross
batch_size = args.batch_size
image_lib = args.data_lib
output_file = os.path.join(args.output_file,'cross'+str(cross))
load_model_name = os.path.join(args.load_model_file, 'cross'+str(cross), 'checkpoint_best.pth')

normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
transform = transforms.Compose([transforms.ToTensor(), normalize])

def get_mask(image, mask):
    (r,g,b) = (image[:,:,0].squeeze(),image[:,:,1].squeeze(),image[:,:,2].squeeze())
    retR, thR = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    retG, thG = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    retB, thB = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    th_temp = cv2.bitwise_and(thR,thG)
    th_temp = cv2.bitwise_and(thB,th_temp)
    th_temp = 255-th_temp
    
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    retH, thH = cv2.threshold(H, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
    #mask = np.zeros(H.shape)
    mask = cv2.bitwise_and(thH, th_temp)
    #mask = cv2.resize(mask,shape,interpolation=cv2.INTER_NEAREST)
    mask[mask<127]=0
    mask[mask>=127]=255
    return mask

def inference(image_temp, model):
    image_temp = Image.fromarray(image_temp)
    image_temp = transform(image_temp)
    image_temp = image_temp.unsqueeze(0)
    image_temp = image_temp.cuda()
    feature, prob = model(image_temp)
    feature = feature.detach().cpu().clone()
    return feature

def proprocess(image, mask, model):
    #print(image.shape)
    time0 = math.floor(image.shape[0]/patch_size)
    time1 = math.floor(image.shape[1]/patch_size)
    #print(image.shape)
    save_map = -1*np.ones((time0,time1))
    index = 0
    features = None
    for i in range(time0):
        x = int(i*patch_size)
        for j in range(time1):
            y = int(j*patch_size)
            mask_temp = mask[x:x+patch_size, y:y+patch_size]
            if np.sum(mask_temp!=0)>(patch_size*patch_size/4):
                img_temp = image[x:x+patch_size, y:y+patch_size,:]
                feature = inference(img_temp, model)
                
                save_map[i,j] = index
                features = torch.cat([features, feature], dim=0) if index!=0 else feature.clone()
                index += 1
                    
    edge_index = None
    flag = False
    direction = [-1, 0, 1]
    for i in range(time0):
        for j in range(time1):
            # is node?
            if save_map[i,j]!=-1:
                node1 = save_map[i,j]
                for n in range(3):
                    for m in range(3):
                        if n == 1 and m == 1:
                            continue
                        # search neighbour
                        if i+direction[n]>=0 and i+direction[n]<time0 and j+direction[m]>=0 \
                        and j+direction[m]<time1 and save_map[i+direction[n],j+direction[m]]!=-1:
                            node2 = save_map[i+direction[n],j+direction[m]]
                            edge_index = torch.Tensor([[node1, node2]]) if edge_index==None \
                            else torch.cat((edge_index,torch.Tensor([[node1, node2]])),0)
                            flag = True
                            #edge_index.append(torch.Tensor([node1, node2]))
    if flag:
        #print(save_map)
        #print(len(patch_list))
        #if edge_index!=None:
        edge_index = edge_index.type(torch.long)
        edge_index = edge_index.t().contiguous()
        edge_index = np.array(edge_index)
        #print(edge_index)
    return features, edge_index, flag 


def process_UID(uid_file, save_file, model):
    image, mask = None, None
    image_names = os.listdir(uid_file)
    for image_name in image_names:
        image = np.asarray(Image.open(os.path.join(uid_file, image_name)))
        mask = get_mask(image, mask)
        features, edge_index, flag = proprocess(image, mask, model)
        features = np.array(features)
        if flag:
            data = {'edge':edge_index, 'feature':features}
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            scio.savemat(os.path.join(save_file, image_name.replace('.jpg', '.mat')), data)
        else:
            print('edge error:',image_name)
    del image
    del mask

def main():
    # load model
    model = Net(BasicBlock,return_fea=True)
    model.cuda()
    model.load_state_dict(torch.load(load_model_name)['state_dict'])
    model.eval()
    print('model load successfully...')
    
    groups = ['TCGA-LUAD/', 'TCGA-LUSC/']
    
    for gr in range(2):
        src = os.path.join(image_lib, groups[gr])
        dst = os.path.join(output_file, groups[gr])
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        uids = os.listdir(src)
        for uid in uids:
            if os.path.exists(os.path.join(dst, uid)):
                continue
            print('current process: '+uid)
            process_UID(os.path.join(src, uid), os.path.join(dst, uid), model)
       

if __name__ == "__main__":
    main()

