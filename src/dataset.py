import os
import cv2
import pdb
import torch
import random
import numpy as np  
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

random.seed(1234)

class DataLoaderTrain(Dataset):
    def __init__(self, dir):
        super(DataLoaderTrain, self).__init__()
        self.rgb_files = sorted(os.listdir(os.path.join(dir, 'rgb')))
        self.rgb_files = [dir + '/rgb/' + name for name in self.rgb_files]
        self.nir_files = sorted(os.listdir(os.path.join(dir, 'nir')))
        self.nir_files = [dir + '/nir/' + name for name in self.nir_files]

    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, index):
        index = index % len(self.rgb_files)
        rgb_path = self.rgb_files[index]
        nir_path = self.nir_files[index]

        rgb, nir = cv2.imread(rgb_path), cv2.imread(nir_path, 0)
        # rgb = cv2.resize(rgb, (1920, 1080))
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB) / 255.
        l_channel = lab[:, :, 0]
        # nir = cv2.resize(nir, (1920, 1080)) / 255.
        nir = nir / 255.
        l_channel, nir = TF.to_tensor(l_channel), TF.to_tensor(nir)

        aug = random.randint(0, 8)
        # Data Augmentations
        if aug==1:
            l_channel, nir = l_channel.flip(1), nir.flip(1)
        elif aug==2:
            l_channel, nir = l_channel.flip(2), nir.flip(2)
        elif aug==3:
            l_channel, nir = torch.rot90(l_channel,dims=(1,2)), torch.rot90(nir, dims=(1,2))
        elif aug==4:
            l_channel, nir = torch.rot90(l_channel,dims=(1,2), k=2), torch.rot90(nir,dims=(1,2), k=2)
        elif aug==5:
            l_channel, nir = torch.rot90(l_channel,dims=(1,2), k=3), torch.rot90(nir,dims=(1,2), k=3)
        elif aug==6:
            l_channel, nir = torch.rot90(l_channel.flip(1),dims=(1,2)), torch.rot90(nir.flip(1),dims=(1,2))
        elif aug==7:
            l_channel, nir = torch.rot90(l_channel.flip(2),dims=(1,2)), torch.rot90(nir.flip(2),dims=(1,2))

        return l_channel.float(), nir.float()