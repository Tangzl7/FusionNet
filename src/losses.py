import pdb
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoss(nn.Module):
    def __init__(self):
        super(DataLoss, self).__init__()
    
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff))
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
        self.window_kernel = torch.tensor([[1/25 for j in range(5)] for i in range(5)], 
                        dtype=torch.float, requires_grad=False).view(1, 1, 5, 5)
        if torch.cuda.is_available():
            self.window_kernel = self.window_kernel.cuda()
    
    def forward(self, x, y):
        smooth_x = F.conv2d(x, self.window_kernel, padding=4)
        smooth_y = F.conv2d(y, self.window_kernel, padding=4)
        diff = smooth_x - smooth_y
        loss = torch.mean(torch.sqrt(diff * diff))
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

    def sobel_conv(self, img):
        edge_detect_x = F.conv2d(img, self.sobel_kernel_x, padding=1)
        edge_detect_y = F.conv2d(img, self.sobel_kernel_y, padding=1)
        edge_detect = torch.sqrt((edge_detect_x * edge_detect_x + edge_detect_y * edge_detect_y)/2)
        return edge_detect
    
    def forward(self, x, n):
        detail_x, detail_n = self.sobel_conv(x), self.sobel_conv(n)
        diff = detail_x - detail_n
        loss = torch.mean(torch.sqrt(diff * diff))
        return loss
