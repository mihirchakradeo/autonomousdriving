# coding: utf-8
import numpy as np
import pandas as pd
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

ROOT_PATH =  'save/experiment1/'
CHECKPOINT_PATH = ROOT_PATH + 'checkpoints/'

try:  
    os.makedirs(CHECKPOINT_PATH)
except OSError:  
    print('Directory already exists!')

# sys.path.insert(0,'./model/')

# import models

device = "cuda"
num_epochs = 500

# #### Helper methods

def get_measurements(file):
    df = pd.read_csv(file, names=['loc1', 'loc2', 'speed', 'c_v', 'c_p', 'c_o', 'other', 'off_road', 'agents', 'throttle', 'steer'], index_col=None)
    speed = pd.to_numeric(df['speed'].str[:-4], downcast='float')
    throttle = pd.to_numeric(df['throttle'].str[10:], downcast='float')
    steer = pd.to_numeric(df['steer'].str[7:], downcast='float')
    return speed.tolist(),throttle.tolist(),steer.tolist()


# #### Data parsing

# measure_path = "/home/bhushan/work/college/Fall18/projects/cv/CARLA/data/measurements/"
# measure_path = "/home/mihir/Downloads/CARLA_0.8.2/PythonClient/_out/measurements/"
# measure_path = '/nfs/bigdisk/bsonawane/carla_dataset/measurements/'
measure_path = '/home/nborude/CARLA_0.8.2/PythonClient/_out/measurements/'

# Loading measurement data
speed_arr = []
throttle_arr = []
steer_arr = []

# for file in glob.glob(path):
# for i in range(97):
for i in range(len(os.listdir(measure_path))):
    with open(measure_path+str(i)+".txt") as file:
        speed,throttle,steer = get_measurements(file)
        speed_arr += (speed)
        throttle_arr += (throttle)
        steer_arr += (steer)
        
# Loading image data
# img_dir_path = "/home/bhushan/work/college/Fall18/projects/cv/CARLA/data/episode*"
# img_dir_path = "/home/mihir/Downloads/CARLA_0.8.2/PythonClient/_out/episode*"

# img_dir_path = '/nfs/bigdisk/bsonawane/carla_dataset/episode*'
img_dir_path = '/home/nborude/CARLA_0.8.2/PythonClient/_out/episode*'
img_path = "/CameraRGB/*.png"


episode_num_arr = []
img_path_arr = []
for directory in sorted(glob.glob(img_dir_path)):
    episode_num = directory[directory.rfind('/')+1:]
    for img in sorted(glob.glob(directory+img_path)):
        episode_num_arr.append(episode_num)
        img = img[img.rfind('/')+1:]
        img_path_arr.append(img)    

# Creating dataframe: episode_number, center_image_path, steer, speed, throttle
df = pd.DataFrame(list(zip(episode_num_arr, img_path_arr, steer_arr, speed_arr, throttle_arr)), columns=['episode_number', 'center','steer','speed','throttle'])
# Writing to CSV
df.to_csv("train.csv")

# #### Dataloader

class CustomDataloader(data.Dataset):
    def __init__(self, df, episode, flag='train'):
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.flag = flag
        self.id_dict = {}
        self.label_dict = {}
#         episode = episode.split("/")[-1]
        self.episode = episode
        self.transform = transforms.ToTensor()
        
        # Preprocess the data here
        for index, row in df.iterrows():
            if row.episode_number not in self.id_dict:
                self.id_dict[row.episode_number] = []
                self.label_dict[row.episode_number] = []

            self.id_dict[row.episode_number].append(row.center)
            self.label_dict[row.episode_number].append([row.steer,row.speed])
        
    def __getitem__(self, index):
        id = self.id_dict[self.episode][index]
        label = torch.tensor(self.label_dict[self.episode][index])
        # images = cv2.imread("/home/mihir/Downloads/CARLA_0.8.2/PythonClient/_out/"+self.episode+"/CameraRGB/"+id)
        images = cv2.imread(img_dir_path+self.episode+"/CameraRGB/"+id)
        images = cv2.imread('/home/nborude/CARLA_0.8.2/PythonClient/_out/'+self.episode+"/CameraRGB/"+id)
        images = self.transform(images)
        return images,label

    def __len__(self):
        return len(self.id_dict[self.episode])

class base_model(nn.Module):
    def __init__(self):
        super(base_model, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride = 2, bias = False),
            nn.ReLU(),

            nn.Conv2d(24, 36, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(36),

            nn.Conv2d(36, 48, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 64, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.Dropout(p=0.35)
        )

        self.linear_net = nn.Sequential(
            nn.Linear(64 * 22 * 15, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 2, bias = False)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 64 * 15 * 22)
        x = self.linear_net(x)
        return x

# model = models.base_model().to(device)
model = base_model().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(train_dirs, CustomDataloader, test_dirs):
    # limit = 5
    for epoch in range(num_epochs):
        c = 0
        t = 0
        loss_arr2 = []
        for directory in (train_dirs):
            # if limit == 0:
            #     break
            # limit -= 1
            ep = directory.split("/")[-1]
            custom_data = CustomDataloader(df, ep, flag='train')
            train_loader = torch.utils.data.DataLoader(dataset=custom_data, batch_size=10)
            #  Training
            loss_arr = []
            
            # print("Episode:", ep)
            for i, (images, labels) in enumerate(train_loader):

                images = images.to(device)
                labels = labels.to(device)
                labels = labels.type(torch.cuda.FloatTensor)
                
                output = model(images)
                # output = output.view(-1)
                loss = loss_fn(output, labels)
                # loss = torch.sum(loss)
                loss_arr.append(loss.item())

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_arr2.append(np.mean(loss_arr))
            print("Loss: ",loss_arr2[-1]," in episode: ",ep)
            sys.stdout.flush()

        print("Epoch: ",epoch, "Train Loss: {:.5f}".format(np.mean(loss_arr2)))
        sys.stdout.flush()
        test(model, test_dirs, CustomDataloader)
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), CHECKPOINT_PATH + 'cnn_base.model')


    return model


def test(model, test_dirs, CustomDataloader):
    loss_arr = []
    
    for directory in (test_dirs):
#         if limit == 0:
#             break
#         limit -= 1
        ep = directory.split("/")[-1]
        custom_data = CustomDataloader(df, ep, flag='test')
        test_loader = torch.utils.data.DataLoader(dataset=custom_data, batch_size=10)
        #  Testing
        c = 0
        t = 0
        
        for i, (images, labels) in enumerate(test_loader):

            images = images.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.cuda.FloatTensor)

            # Forward
            output = model(images)
            output = output.view(-1)
            loss = loss_fn(output, labels)
            loss_arr.append(loss.item())

    print("Test Loss: {:.5f}".format(np.mean(loss_arr)))
    sys.stdout.flush()

# custom_data = CustomDataloader(df, "episode_0000", flag='train')
# dirs = sorted(glob.glob("/nfs/bigdisk/bsonawane/carla_dataset/episode*"))
# dirs = sorted(glob.glob("/home/mihir/Downloads/CARLA_0.8.2/PythonClient/_out/episode*"))
dirs = sorted(glob.glob("/home/nborude/CARLA_0.8.2/PythonClient/_out/episode*"))


train_dirs = dirs[:-5]
test_dirs = dirs[-5:]

model = train(train_dirs, CustomDataloader, test_dirs)
# test(model, test_dirs, CustomDataloader)
