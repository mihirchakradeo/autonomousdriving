# coding: utf-8
import numpy as np
import cv2
#import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from sklearn.model_selection import train_test_split

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import glob

import sys
import os

ROOT_PATH =  'save/experiment_all_controls/'
CHECKPOINT_PATH = ROOT_PATH + 'checkpoints/'

try:  
    os.makedirs(CHECKPOINT_PATH)
except OSError:  
    print('Directory already exists!')

from models_2 import immitation_model
from data_loader_2 import H5_DataLoader

device = "cuda"
num_epochs = 200

model = immitation_model().to(device)
model.load_state_dict(torch.load(ROOT_PATH+'checkpoints_full_data_25_epochs/cnn_base.model'))
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(train_dirs, CustomDataloader, test_dirs):
    # limit = 5
    custom_data = CustomDataloader(train_dirs)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data, batch_size=10)
    print('Dataset loaded:' , len(train_loader))
    for epoch in range(num_epochs):
        loss_arr2 = []
        
        #  Training
        loss_arr = []
        
        for i, (images, labels) in enumerate(train_loader):
            # print(images.shape, labels.shape)
            images = images.to(device)
            labels = torch.tensor(labels).to(device)
            # labels = torch.reshape(labels, (-1, 1))
            # labels = labels.type(torch.cuda.FloatTensor)
            
            out_steer, out_speed, out_brake, out_acceleration = model(images)
            #output = output.view(-1)
            # labels: bx4
            loss_steer = loss_fn(out_steer, labels[:,0])
            loss_speed = loss_fn(out_speed, labels[:, 1])
            loss_brake = loss_fn(out_brake, labels[:, 2])
            loss_acceleration = loss_fn(out_acceleration, labels[:, 3])
            loss = loss_steer + loss_speed + loss_brake + loss_acceleration

            # loss = torch.sum(loss)
            loss_arr.append(loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_arr2.append(np.mean(loss_arr))
            # print("Loss: ",loss_arr2[-1])
            sys.stdout.flush()

        print("Epoch: ",epoch, "Train Loss: {:.5f}".format(np.mean(loss_arr2)))
        sys.stdout.flush()
        test(model, test_dirs, CustomDataloader)
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), CHECKPOINT_PATH + 'cnn_base.model')
    return model


def test(model, test_dirs, CustomDataloader):
    loss_arr = []
    
    custom_data = CustomDataloader(test_dirs)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data, batch_size=10)
    #  Testing
    
    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        labels = labels.to(device)
        out_steer, out_speed, out_brake, out_acceleration = model(images)
        #output = output.view(-1)
        # labels: bx4
        loss_steer = loss_fn(out_steer, labels[:,0])
        loss_speed = loss_fn(out_speed, labels[:, 1])
        loss_brake = loss_fn(out_brake, labels[:, 2])
        loss_acceleration = loss_fn(out_acceleration, labels[:, 3])
        loss = loss_steer + loss_speed + loss_brake + loss_acceleration


        # Forward
        loss_arr.append(loss.item())

    print("Test Loss: {:.5f}".format(np.mean(loss_arr)))
    sys.stdout.flush()


train_dirs = '/nfs/bigdisk/bsonawane/Carla_Imitation_Learning/AgentHuman/SeqTrain/'
test_dirs = '/nfs/bigdisk/bsonawane/Carla_Imitation_Learning/AgentHuman/SeqVal/'

model = train(train_dirs, H5_DataLoader, test_dirs)
test(model, test_dirs, H5_DataLoader)
