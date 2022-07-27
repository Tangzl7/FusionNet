import os
import cv2
import torch
import random
import numpy as np  
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class DataLoaderTrain(Dataset):
    def __init__(self, dir):
        super(DataLoaderTrain, self).__init__()
        self.rgb_files = sorted(dir + '/rgb/' + os.listdir(os.path.join(dir, 'rgb')))
        self.nir_files = sorted(dir + '/nir/' + os.listdir(os.path.join(dir, 'nir')))

    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, index):
        index = index % len(self.rgb_files)

        rgb_path = self.rgb_files[index]
        nir_path = self.nir_files[index]

        rgb = cv2.imread(rgb_path) / 255.
        nir = cv2.imread(nir_path, 0) / 255.
        rgb, nir = np.transpose(rgb, (2, 0, 1)), np.expand_dims(nir, 0)
        high_reflection_nir = np.expand_dims(np.where(nir>=100/255., 1., 0.), 0)
        high_reflection_bgr = np.expand_dims(1 - high_reflection_nir)

        rgb, nir = TF.to_tensor(rgb), TF.to_tensor(nir)
        high_reflection_bgr, high_reflection_nir = TF.to_tensor(high_reflection_bgr), TF.to_tensor(high_reflection_nir)

        aug = random.randint(0, 8)
        # Data Augmentations
        if aug==1:
            rgb, nir = rgb.flip(1), nir.flip(1)
            high_reflection_bgr, high_reflection_nir = high_reflection_bgr.flip(1), high_reflection_nir.flip(1)
        elif aug==2:
            rgb, nir = rgb.flip(2), nir.flip(2)
            high_reflection_bgr, high_reflection_nir = high_reflection_bgr.flip(2), high_reflection_nir.flip(2)
        elif aug==3:
            rgb, nir = torch.rot90(rgb,dims=(1,2)), torch.rot90(nir, dims=(1,2))
            high_reflection_bgr, high_reflection_nir = torch.rot90(high_reflection_bgr,dims=(1,2)), torch.rot90(high_reflection_nir, dims=(1,2))
        elif aug==4:
            rgb, nir = torch.rot90(rgb,dims=(1,2), k=2), torch.rot90(nir,dims=(1,2), k=2)
            high_reflection_bgr, high_reflection_nir = torch.rot90(high_reflection_bgr,dims=(1,2), k=2), torch.rot90(high_reflection_nir,dims=(1,2), k=2)
        elif aug==5:
            rgb, nir = torch.rot90(rgb,dims=(1,2), k=3), torch.rot90(nir,dims=(1,2), k=3)
            high_reflection_bgr, high_reflection_nir = torch.rot90(high_reflection_bgr,dims=(1,2), k=3), torch.rot90(high_reflection_nir,dims=(1,2), k=3)
        elif aug==6:
            rgb, nir = torch.rot90(rgb.flip(1),dims=(1,2)), torch.rot90(nir.flip(1),dims=(1,2))
            high_reflection_bgr, high_reflection_nir = torch.rot90(high_reflection_bgr.flip(1),dims=(1,2)), torch.rot90(high_reflection_nir.flip(1),dims=(1,2))
        elif aug==7:
            rgb, nir = torch.rot90(rgb.flip(2),dims=(1,2)), torch.rot90(nir.flip(2),dims=(1,2))
            high_reflection_bgr, high_reflection_nir = torch.rot90(high_reflection_bgr.flip(2),dims=(1,2)), torch.rot90(high_reflection_nir.flip(2),dims=(1,2))

        return rgb, nir, high_reflection_bgr, high_reflection_nir