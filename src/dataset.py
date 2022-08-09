import os
import cv2
import pdb
import torch
import random
import numpy as np  
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

random.seed(1234)

def otsu(img):
    img = np.uint8(255. * img)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist / (img.shape[0] * img.shape[1])
    thr, tmp_max = -1, 0
    for i in range(256):
        w0, w1, u0, u1 = 0, 0, 0, 0
        for j in range(256):
            if i <= j:
                w0 += hist[j]
                u0 += hist[j] * j
            else:
                w1 += hist[j]
                u1 += hist[j] * j
        if w0 == 0 or w1 == 0:
            continue
        u0, u1 = u0 / w0, u1 / w1
        var = w0 * w1 * (u0 - u1) * (u0 - u1)
        thr = np.where(var > tmp_max, i, thr)
        tmp_max = max(tmp_max, var)

    high_reflection = np.where(img >= thr, 1., 0.)
    return high_reflection

class DataLoaderTrain(Dataset):
    def __init__(self, dir):
        super(DataLoaderTrain, self).__init__()
        self.rgb_files = sorted(os.listdir(os.path.join(dir, 'rgb')))
        self.rgb_files = [dir + '/rgb/' + name for name in self.rgb_files]
        self.nir_files = sorted(os.listdir(os.path.join(dir, 'nir')))
        self.nir_files = [dir + '/new_nir/' + name for name in self.nir_files]
        self.gt_files = sorted(os.listdir(os.path.join(dir, 'gt')))
        self.gt_files = [dir + '/gt/' + name for name in self.gt_files]

    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, index):
        index = index % len(self.rgb_files)
        rgb_path = self.rgb_files[index]
        nir_path = self.nir_files[index]
        gt_path = self.gt_files[index]

        rgb, nir, gt = cv2.imread(rgb_path), cv2.imread(nir_path, 0), cv2.imread(gt_path)
        # rgb = cv2.resize(rgb, (1920, 1080))
        lab_rgb, lab_gt = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB) / 255., cv2.cvtColor(gt, cv2.COLOR_BGR2LAB) / 255.
        l_0, l_1 = lab_rgb[:, :, 0], lab_gt[:, :, 0]
        # nir = cv2.resize(nir, (1920, 1080)) / 255.
        nir = nir / 255.
        l_0, nir, l_1 = TF.to_tensor(l_0), TF.to_tensor(nir), TF.to_tensor(l_1)

        aug = random.randint(0, 8)
        # Data Augmentations
        # if aug==1:
        #     l_channel, nir = l_channel.flip(1), nir.flip(1)
        # elif aug==2:
        #     l_channel, nir = l_channel.flip(2), nir.flip(2)
        # elif aug==3:
        #     l_channel, nir = torch.rot90(l_channel,dims=(1,2)), torch.rot90(nir, dims=(1,2))
        # elif aug==4:
        #     l_channel, nir = torch.rot90(l_channel,dims=(1,2), k=2), torch.rot90(nir,dims=(1,2), k=2)
        # elif aug==5:
        #     l_channel, nir = torch.rot90(l_channel,dims=(1,2), k=3), torch.rot90(nir,dims=(1,2), k=3)
        # elif aug==6:
        #     l_channel, nir = torch.rot90(l_channel.flip(1),dims=(1,2)), torch.rot90(nir.flip(1),dims=(1,2))
        # elif aug==7:
        #     l_channel, nir = torch.rot90(l_channel.flip(2),dims=(1,2)), torch.rot90(nir.flip(2),dims=(1,2))

        return l_0.float(), nir.float(), l_1.float()


class DataLoaderForJFC(Dataset):
    def __init__(self, dir):
        super(DataLoaderForJFC, self).__init__()
        self.rgb_files = sorted(os.listdir(os.path.join(dir, 'rgb')))
        self.rgb_files = [dir + '/rgb/' + name for name in self.rgb_files]
        self.nir_files = sorted(os.listdir(os.path.join(dir, 'nir')))
        self.nir_files = [dir + '/nir/' + name for name in self.nir_files]
        self.gt_files = sorted(os.listdir(os.path.join(dir, 'gt')))
        self.gt_files = [dir + '/gt/' + name for name in self.gt_files]

    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, index):
        index = index % len(self.rgb_files)
        rgb_path = self.rgb_files[index]
        nir_path = self.nir_files[index]
        gt_path = self.gt_files[index]

        rgb, nir, gt = cv2.imread(rgb_path)/255., cv2.imread(nir_path, 0)/255., cv2.imread(gt_path)/255.
        nir_mask = otsu(nir)
        rgb, nir, gt, nir_mask = TF.to_tensor(rgb), TF.to_tensor(nir), TF.to_tensor(gt), TF.to_tensor(nir_mask)

        return rgb.float(), nir.float(), gt.float(), nir_mask.float()
