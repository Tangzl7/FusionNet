import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import pdb
import numpy as np
from PIL import Image

from dataset import otsu

class CALayer(nn.Module):
    def __init__(self, channel, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv_ca(out)
        return x * out

class Sample(nn.Module):
    def __init__(self, width, height):
        super(Sample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(size=(width, height), mode='bilinear', align_corners=False))

    def forward(self, x):
        out = self.down(x)
        return out

class DeepConvWeigthNet(nn.Module):
    def __init__(self, cuda_using=True):
        super(DeepConvWeigthNet, self).__init__()
        # self.kernels_1 = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 
        #                 dtype=torch.float).view(1, 1, 3, 3)
        self.kernels_1 = torch.tensor([[1]], 
                        dtype=torch.float).view(1, 1, 1, 1)
        self.kernels_2 = torch.tensor([[1/25 for j in range(5)] for i in range(5)], 
                        dtype=torch.float).view(1, 1, 5, 5)
        self.kernels_3 = torch.tensor([[1/225 for j in range(15)] for i in range(15)], 
                        dtype=torch.float).view(1, 1, 15, 15)
        self.kernels_4 = torch.tensor([[1/625 for j in range(25)] for i in range(25)], 
                        dtype=torch.float).view(1, 1, 25, 25)
        if torch.cuda.is_available() and cuda_using:
            self.kernels_1 = self.kernels_1.cuda()
            self.kernels_2 = self.kernels_2.cuda()
            self.kernels_3 = self.kernels_3.cuda()
            self.kernels_4 = self.kernels_4.cuda()
        
        self.module_body, self.module_head_1, self.module_head_2, self.module_head_3 = [], [], [], []
        self.module_body.append(nn.Conv2d(3, 32, 3, padding='same'))
        self.module_body.append(nn.PReLU())
        self.module_body.append(nn.Conv2d(32, 64, 3, padding='same'))
        self.module_body.append(nn.PReLU())
        self.module_body.append(nn.Conv2d(64, 32, 3, padding='same'))
        self.module_body.append(nn.PReLU())

        self.module_head_1.append(nn.Conv2d(32, 4, 3, padding='same'))
        self.module_head_1.append(CALayer(4))
        self.module_head_1.append(nn.Softmax2d())

        self.module_head_2.append(nn.Conv2d(32, 4, 3, padding='same'))
        self.module_head_2.append(CALayer(4))
        self.module_head_2.append(nn.Softmax2d())

        self.module_head_3.append(nn.Conv2d(32, 4, 3, padding='same'))
        self.module_head_3.append(CALayer(4))
        self.module_head_3.append(nn.Softmax2d())

        self.module_body = nn.Sequential(*self.module_body)
        self.module_head_1 = nn.Sequential(*self.module_head_1)
        self.module_head_2 = nn.Sequential(*self.module_head_2)
        self.module_head_3 = nn.Sequential(*self.module_head_3)
        
    def forward(self, x):
        body_out = self.module_body(x)
        head_out_1 = self.module_head_1(body_out)
        head_out_2 = self.module_head_2(body_out)
        head_out_3 = self.module_head_3(body_out)
        h1_r1, h1_r2, h1_r3, h1_r4 = torch.split(head_out_1, 1, dim=1)
        h2_r1, h2_r2, h2_r3, h2_r4 = torch.split(head_out_2, 1, dim=1)
        h3_r1, h3_r2, h3_r3, h3_r4 = torch.split(head_out_3, 1, dim=1)

        conv_map_1_1 = F.conv2d(x, self.kernels_1.repeat(3, 1, 1, 1), padding=0, groups=3)
        conv_map_1_2 = F.conv2d(x, self.kernels_2.repeat(3, 1, 1, 1), padding=2, groups=3)
        conv_map_1_3 = F.conv2d(x, self.kernels_3.repeat(3, 1, 1, 1), padding=7, groups=3)
        conv_map_1_4 = F.conv2d(x, self.kernels_4.repeat(3, 1, 1, 1), padding=12, groups=3)
        out_1 = h1_r1 * conv_map_1_1 + h1_r2 * conv_map_1_2 + h1_r3 * conv_map_1_3 + h1_r4 * conv_map_1_4

        conv_map_2_1 = F.conv2d(out_1, self.kernels_1.repeat(3, 1, 1, 1), padding=0, groups=3)
        conv_map_2_2 = F.conv2d(out_1, self.kernels_2.repeat(3, 1, 1, 1), padding=2, groups=3)
        conv_map_2_3 = F.conv2d(out_1, self.kernels_3.repeat(3, 1, 1, 1), padding=7, groups=3)
        conv_map_2_4 = F.conv2d(out_1, self.kernels_4.repeat(3, 1, 1, 1), padding=12, groups=3)
        out_2 = h2_r1 * conv_map_2_1 + h2_r2 * conv_map_2_2 + h2_r3 * conv_map_2_3 + h2_r4 * conv_map_2_4

        conv_map_3_1 = F.conv2d(out_2, self.kernels_1.repeat(3, 1, 1, 1), padding=0, groups=3)
        conv_map_3_2 = F.conv2d(out_2, self.kernels_2.repeat(3, 1, 1, 1), padding=2, groups=3)
        conv_map_3_3 = F.conv2d(out_2, self.kernels_3.repeat(3, 1, 1, 1), padding=7, groups=3)
        conv_map_3_4 = F.conv2d(out_2, self.kernels_4.repeat(3, 1, 1, 1), padding=12, groups=3)
        out_3 = h3_r1 * conv_map_3_1 + h3_r2 * conv_map_3_2 + h3_r3 * conv_map_3_3 + h3_r4 * conv_map_3_4

        return out_3

class NirFeatExtrator(nn.Module):
    def __init__(self):
        super(NirFeatExtrator, self).__init__()
        self.extractor = []
        self.extractor.append(nn.Conv2d(1, 96, 9, padding=2))
        self.extractor.append(nn.ReLU())
        self.extractor.append(nn.Conv2d(96, 48, 1, padding=2))
        self.extractor.append(nn.ReLU())
        self.extractor.append(nn.Conv2d(48, 3, 5, padding=2))
        self.extractor = nn.Sequential(*self.extractor)
    
    def forward(self, x, mask):
        out = self.extractor(x)
        out = out * mask
        return out


class SmoothingNet(nn.Module):
    def __init__(self, cuda_using=True):
        super(SmoothingNet, self).__init__()
        self.conv_weight_net = DeepConvWeigthNet(cuda_using)
        self.nir_feat_extractor = NirFeatExtrator()

    def forward(self, x, y, mask):
        denoised = self.conv_weight_net(x)
        nir_detail = self.nir_feat_extractor(y, mask)
        fusion = denoised + nir_detail
        out = torch.clamp(fusion, 0., 1.)
        return  denoised, out


def generate_data():
    net = SmoothingNet(False)
    net.load_state_dict(torch.load('./snapshots/joint_filter_cnn.pth'))
    dir = '../data'
    rgb_files = sorted(os.listdir(os.path.join(dir, 'rgb')))
    rgb_files = [dir + '/rgb/' + name for name in rgb_files]
    nir_files = sorted(os.listdir(os.path.join(dir, 'nir')))
    nir_files = [dir + '/nir/' + name for name in nir_files]
    for i in range(len(rgb_files)):
        bgr_path, nir_path = rgb_files[i], nir_files[i]
        bgr, nir = cv2.imread(bgr_path) / 255., cv2.imread(nir_path, 0) / 255.
        nir_mask = TF.to_tensor(otsu(nir))
        bgr, nir = TF.to_tensor(bgr), TF.to_tensor(nir)
        bgr, nir, nir_mask = torch.unsqueeze(bgr.float(), 0), torch.unsqueeze(nir.float(), 0), torch.unsqueeze(nir_mask.float(), 0)
        _, out = net(bgr, nir, nir_mask)
        out = torch.squeeze(out, 0).detach().numpy()
        out = np.transpose(out, (1, 2, 0)) * 255.
        out = np.clip(out, 0, 255)
        cv2.imwrite(bgr_path.replace('/rgb', '/generate_step_1'), np.uint8(out))

if __name__=="__main__":
    net = SmoothingNet(False)
    net.load_state_dict(torch.load('./snapshots/smoothing_cnn.pth'))
    bgr, nir = cv2.imread('../data/original_data/0001_rgb.jpg') / 255., cv2.imread('../data/original_data/0001_nir.jpg', 0) / 255.
    nir_mask = TF.to_tensor(otsu(nir))
    bgr, nir = TF.to_tensor(bgr), TF.to_tensor(nir)
    bgr, nir, nir_mask = torch.unsqueeze(bgr.float(), 0), torch.unsqueeze(nir.float(), 0), torch.unsqueeze(nir_mask.float(), 0)
    _, out = net(bgr, nir, nir_mask)
    out = torch.squeeze(out, 0).detach().numpy()
    out = np.transpose(out, (1, 2, 0)) * 255.
    out = np.clip(out, 0, 255)
    cv2.imwrite('smt_tmp.png', np.uint8(out))

