import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import pdb
import numpy as np
from PIL import Image

from dataset import otsu

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

if __name__ == '__main__':
    net = JointFilterCNN()
    net.load_state_dict(torch.load('./snapshots/joint_filter_cnn.pth'))
    bgr, nir = cv2.imread('../data/original_data/0003_rgb.jpg') / 255., cv2.imread('../data/original_data/0003_nir.jpg', 0) / 255.
    nir_mask = TF.to_tensor(otsu(nir))
    bgr, nir = TF.to_tensor(bgr), TF.to_tensor(nir)
    bgr, nir, nir_mask = torch.unsqueeze(bgr.float(), 0), torch.unsqueeze(nir.float(), 0), torch.unsqueeze(nir_mask.float(), 0)
    out = net(bgr, nir, nir_mask)
    out = torch.squeeze(out, 0).detach().numpy()
    out = np.transpose(out, (1, 2, 0)) * 255.
    print(out.shape)
    cv2.imwrite('jfc_tmp.png', np.uint8(out))
