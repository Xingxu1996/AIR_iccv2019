# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:13:12 2018

@author: yxx_h
"""

import torch.utils.data as data
import torch
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import os.path
import glob
import numpy as np
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, root, datatxt, train=True, transform=None, target_transform=None):
        self.train = train
        fh=open(root+datatxt,'r')
        imgs=[]
        train_labels=[]
        train_labels2 = []
        train_data=[]
        for line in fh:
            line=line.rstrip()
            words=line.split()
            train_data.append((words[0]))
            imgs.append((words[0], int(words[1]), int(words[2])))
            train_labels.append((int(words[1])))
            train_labels2.append((int(words[2])))
        self.train_data = train_data
        self.train_labels = torch.LongTensor(train_labels)
        self.train_labels2 = torch.LongTensor(train_labels2)
        self.imgs = imgs
        self.transform=transform
        self.target_transform = target_transform
    def __getitem__(self,index):
        fn, label, label2=self.imgs[index]
        root='/home/ubuntu2/sdy/dataset/fi/'
        img=Image.open(root+fn).convert('RGB')
        
        if self.transform is not None:
            img=self.transform(img)
        temp = np.array((1,))
        temp2 = np.array((1,))
        temp[0] = label
        temp2[0] = label2
        label = torch.LongTensor(temp)[0]
        label2 = torch.LongTensor(temp2)[0]
        return img, label, label2
       
    def __len__(self):
        return len(self.imgs)

