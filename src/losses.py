import pdb
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoss(nn.Module):
    def __init__(self):
        super(DataLoss, self).__init__()
    
    def forward(self, x, y):
        # pdb.set_trace()
        diff = x - y
        loss = torch.mean(diff * diff)
        return loss

class AreaEdgeLoss(nn.Module):
    def __init__(self):
        super(AreaEdgeLoss, self).__init__()
        self.sobel_kernel_x = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                        dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                        dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        if torch.cuda.is_available():
            self.sobel_kernel_x = self.sobel_kernel_x.cuda()
            self.sobel_kernel_y = self.sobel_kernel_y.cuda()

    def sobel_conv(self, img, channels):
        edge_detect_x = F.conv2d(img, self.sobel_kernel_x.repeat(1, channels, 1, 1), padding=1)
        edge_detect_y = F.conv2d(img, self.sobel_kernel_y.repeat(1, channels, 1, 1), padding=1)
        edge_detect = torch.sqrt((edge_detect_x * edge_detect_x + edge_detect_y * edge_detect_y)/2)
        return edge_detect
    
    def forward(self, x, v, n, r_v, r_n):
        # pdb.set_trace()
        detail_x, detail_v, detail_n = self.sobel_conv(x, 3), self.sobel_conv(v, 3), self.sobel_conv(n, 1)
        loss_1 = torch.mean(torch.sqrt(r_v * (detail_x - detail_v) * (detail_x - detail_v)))
        loss_2 = torch.mean(torch.sqrt(r_n * (detail_x - detail_n) * (detail_x - detail_n)))
        loss = loss_1 + loss_2
        return loss_2


class DataWindowLoss(nn.Module):
    def __init__(self):
        super(DataWindowLoss, self).__init__()
        self.window_kernel = torch.tensor([[1/49 for j in range(7)] for i in range(7)], 
                        dtype=torch.float, requires_grad=False).view(1, 1, 7, 7)
        if torch.cuda.is_available():
            self.window_kernel = self.window_kernel.cuda()
    
    def forward(self, x, y):
        smooth_x = F.conv2d(x, self.window_kernel.repeat(1, 3, 1, 1), padding=3)
        smooth_y = F.conv2d(y, self.window_kernel.repeat(1, 3, 1, 1), padding=3)
        diff = smooth_x - smooth_y
        loss = torch.mean(torch.sqrt(diff * diff + 1e-6))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_kernel_x = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                        dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                        dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        if torch.cuda.is_available():
            self.sobel_kernel_x = self.sobel_kernel_x.cuda()
            self.sobel_kernel_y = self.sobel_kernel_y.cuda()

    def sobel_conv(self, img, channel):
        edge_detect_x = F.conv2d(img, self.sobel_kernel_x.repeat(1, channel, 1, 1), padding=1)
        edge_detect_y = F.conv2d(img, self.sobel_kernel_y.repeat(1, channel, 1, 1), padding=1)
        edge_detect = torch.sqrt((edge_detect_x * edge_detect_x + edge_detect_y * edge_detect_y)/2 + 1e-6)
        return edge_detect
    
    def forward(self, x, n, mask, channel=3):
        detail_x, detail_n = self.sobel_conv(x, channel), self.sobel_conv(n, channel)
        # detail_x, detail_n = detail_x * mask, detail_n * mask
        diff = detail_x - detail_n
        loss = torch.mean(torch.abs(diff))
        return loss


class SmoothingLoss(nn.Module):
    def __init__(self):
        super(SmoothingLoss, self).__init__()
        self.smoothing_kernel = torch.tensor([[-1 for i in range(9)] for j in range(9)], dtype=torch.float, requires_grad=False)
        self.smoothing_kernel[4][4] = 80
        self.smoothing_kernel = self.smoothing_kernel.view(1, 1, 9, 9)
        if torch.cuda.is_available():
            self.smoothing_kernel = self.smoothing_kernel.cuda()

    def forward(self, x, gt, channel=3):
        x_out = F.conv2d(x, self.smoothing_kernel.repeat(1, channel, 1, 1), padding=4)
        gt_out = F.conv2d(gt, self.smoothing_kernel.repeat(1, channel, 1, 1), padding=4)
        weight = 1 - torch.max(gt, dim=1).values
        x_out = x_out * weight
        gt_out = gt_out * weight
        loss = torch.mean(torch.abs(x_out - gt_out))
        return loss