from turtle import forward
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

class DeepConvWeigthNet(nn.Module):
    def __init__(self):
        super(DeepConvWeigthNet, self).__init__()
        self.kernels_1 = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                        dtype=torch.float).view(1, 1, 3, 3)
        self.kernels_2 = torch.tensor([[1]], 
                        dtype=torch.float).view(1, 1, 1, 1)
        self.kernels_3 = torch.tensor([[1/9 for j in range(3)] for i in range(3)], 
                        dtype=torch.float).view(1, 1, 3, 3)
        self.kernels_4 = torch.tensor([[1/25 for j in range(5)] for i in range(5)], 
                        dtype=torch.float).view(1, 1, 5, 5)
        self.kernels_5 = torch.tensor([[1/49 for j in range(7)] for i in range(7)], 
                        dtype=torch.float).view(1, 1, 7, 7)
        
        self.module_body = []
        self.module_body.append(nn.Conv2d(3, 32, 3, padding='same'))
        self.module_body.append(nn.PReLU())
        self.module_body.append(nn.Conv2d(32, 64, 3, padding='same'))
        self.module_body.append(nn.PReLU())
        self.module_body.append(nn.Conv2d(64, 32, 3, padding='same'))
        self.module_body.append(nn.PReLU())
        self.module_body.append(nn.Conv2d(32, 5, 3, padding='same'))
        self.module_body.append(CALayer(5))
        self.module_body.append(nn.Softmax2d())
        self.module_body = nn.Sequential(*self.module_body)
        
    def forward(self, x):
        # pdb.set_trace()
        conv_weight = self.module_body(x)
        r1, r2, r3, r4, r5 = torch.split(conv_weight, 1, dim=1)

        conv_map_1 = F.conv2d(x, self.kernels_1.repeat(3, 1, 1, 1), padding=1, groups=3)
        conv_map_2 = F.conv2d(x, self.kernels_2.repeat(3, 1, 1, 1), padding=0, groups=3)
        conv_map_3 = F.conv2d(x, self.kernels_3.repeat(3, 1, 1, 1), padding=1, groups=3)
        conv_map_4 = F.conv2d(x, self.kernels_4.repeat(3, 1, 1, 1), padding=2, groups=3)
        conv_map_5 = F.conv2d(x, self.kernels_5.repeat(3, 1, 1, 1), padding=3, groups=3)

        out = r1 * conv_map_1 + r2 * conv_map_2 + r3 * conv_map_3 + r4 * conv_map_4 + r5 * conv_map_5
        # print(conv_map_3.shape)
        return out


class JointFilterCNN(nn.Module):
    def __init__(self):
        super(JointFilterCNN, self).__init__()
        self.nir_thr = torch.nn.Parameter(torch.FloatTensor([0.4]), requires_grad=True)
        self.cnn_t, self.cnn_g, self.cnn_f = [], [], []

        self.cnn_t.append(nn.Conv2d(3, 96, 9, padding=2))
        self.cnn_t.append(nn.ReLU())
        self.cnn_t.append(nn.Conv2d(96, 48, 1, padding=2))
        self.cnn_t.append(nn.ReLU())
        self.cnn_t.append(nn.Conv2d(48, 3, 5, padding=2))

        self.cnn_g.append(nn.Conv2d(1, 96, 9, padding=2))
        self.cnn_g.append(nn.ReLU())
        self.cnn_g.append(nn.Conv2d(96, 48, 1, padding=2))
        self.cnn_g.append(nn.ReLU())
        self.cnn_g.append(nn.Conv2d(48, 3, 5, padding=2))

        self.cnn_f.append(nn.Conv2d(3, 64, 9, padding=2))
        self.cnn_f.append(nn.ReLU())
        self.cnn_f.append(nn.Conv2d(64, 32, 1, padding=2))
        self.cnn_f.append(nn.ReLU())
        self.cnn_f.append(nn.Conv2d(32, 3, 5, padding=2))

        self.cnn_t = nn.Sequential(*self.cnn_t)
        self.cnn_g = nn.Sequential(*self.cnn_g)
        self.cnn_f = nn.Sequential(*self.cnn_f)

    def forward(self, x, y, mask):
        t_out = self.cnn_t(x)
        g_out = self.cnn_g(y) * mask
        out = t_out + g_out
        out = self.cnn_f(out) + x
        out = torch.clamp(out, 0., 1.)
        return out

def conv_operator(img, in_channels=1):
    kernels_3 = torch.tensor([[1/9 for j in range(3)] for i in range(3)], 
                        dtype=torch.float).view(1, 1, 3, 3)

    dx = F.conv2d(img, kernels_3.repeat(1, in_channels, 1, 1), stride=1, padding=1,)

    return dx.squeeze(0).numpy().transpose(1, 2, 0)


if __name__=="__main__":
    net = DeepConvWeigthNet()

    img = cv2.imread('../data/rgb/0_rgb.jpg') / 255.
    img = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    y = net(img)
    # pdb.set_trace()
    y = y.squeeze(0).detach().numpy().transpose(1, 2, 0) * 255.
    y = np.clip(y, 0, 255)
    # y = conv_operator(img, 3)
    cv2.imwrite('jfc_tmp.png', np.uint8(y))