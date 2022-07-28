
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
        super(AreaEdgeLoss).__init__()
        self.sobel_kernel_x = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                        dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                        dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        if torch.cuda.is_available():
            self.sobel_kernel_x = self.sobel_kernel_x.cuda()
            self.sobel_kernel_y = self.sobel_kernel_y.cuda()

    def sobel_conv(self, img):
        edge_detect_x = F.conv2d(img, self.sobel_kernel_x.repeat(1, 3, 1, 1), padding=1)
        edge_detect_y = F.conv2d(img, self.sobel_kernel_y.repeat(1, 3, 1, 1), padding=1)
        edge_detect = torch.sqrt((edge_detect_x * edge_detect_x + edge_detect_y * edge_detect_y)/2)
        return edge_detect
    
    def forward(self, x, v, n, r_v, r_n):
        detail_x, detail_v, detail_n = self.sobel_conv(x), self.sobel_conv(v), self.sobel_conv(n)
        loss_1 = torch.mean(torch.sqrt(r_v * (detail_x - detail_v) * (detail_x - detail_v)))
        loss_2 = torch.mean(torch.sqrt(r_n * (detail_x - detail_n) * (detail_x - detail_n)))
        loss = loss_1 + loss_2
        return loss        
