import torch
from torch.utils import data
import h5py
import glob
import torchvision.transforms as transforms

class H5_DataLoader(data.Dataset):
    def __init__(self, path):
        data = sorted(glob.glob(path+"*.h5"))

        self.images = []
        self.labels = []

        for file in data:
            try:
                image_file = h5py.File(file, 'r')
                for i in range(200):
                    self.images.append(image_file['rgb'][i])
                    self.labels.append(image_file['targets'][i][0])
            except:
                continue
            
        
    def __getitem__(self, index):
        transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
        image = transform(self.images[index])
        label = torch.tensor(self.labels[index], dtype = torch.float32)
        # print(image.shape, label.shape)
        
        # Steer, Gas, Brake, Speed
        # new_labels = [label[0], label[2], label[3], label[4]]
        return image, label
        
    def __len__(self):
        return len(self.labels)