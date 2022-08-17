from tkinter.messagebox import NO
import cv2
import pdb
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
 
 
def nn_conv2d(im):
  # 用nn.Conv2d定义卷积操作
  conv_op = nn.Conv2d(1, 1, 3, bias=False)
  # 定义sobel算子参数
  sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
  # 将sobel算子转换为适配卷积操作的卷积核
  sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
  # 给卷积操作的卷积核赋值
  conv_op.weight.data = torch.from_numpy(sobel_kernel)
  # 对图像进行卷积操作
  edge_detect = conv_op(Variable(im))
  # 将输出转换为图片格式
  edge_detect = edge_detect.squeeze().detach().numpy()
  return edge_detect
 
def functional_conv2d(im):
  sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32') #
  sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32') #
  sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
  sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))
  weight_x = Variable(torch.from_numpy(sobel_kernel_x).repeat(1,3,1,1))
  weight_y = Variable(torch.from_numpy(sobel_kernel_y).repeat(1,3,1,1))
  edge_detect_x = F.conv2d(Variable(im), weight_x, groups=3)
  edge_detect_y = F.conv2d(Variable(im), weight_y, groups=3)
  print(weight_x.shape)
  edge_detect = torch.sqrt(edge_detect_x * edge_detect_x + edge_detect_y * edge_detect_x)
  edge_detect = edge_detect.squeeze().detach().numpy()
  return edge_detect
 
sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
laplace = torch.tensor([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
avgpool = torch.tensor([[1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
tmp_kernel = torch.tensor([[-1 for i in range(21)] for j in range(21)], dtype=torch.float, requires_grad=False)
tmp_kernel[10][10] = 440
tmp_kernel = tmp_kernel.view(1, 1, 21, 21)

def conv_operator(filename, in_channels=1):
    img = cv2.imread(filename)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    dx = F.conv2d(x, tmp_kernel.repeat(1, in_channels, 1, 1), stride=1, padding=10,)
    dy = F.conv2d(x, sobel_y.repeat(1, in_channels, 1, 1), stride=1, padding=1,)
    out = torch.sqrt((dx * dx + dy * dy)/2)
    out = out.squeeze(0).numpy().transpose(1, 2, 0)

    return img, dx.squeeze(0).numpy().transpose(1, 2, 0)


if __name__=="__main__":
    img_name = '../data/original_data/0003_nir.jpg'

    img, y = conv_operator(img_name, 3)
    cv2.imwrite('edge5.png', y)
    # pdb.set_trace()
    # x = torch.arange(0, 1*1*5*5).float()
    # x = x.view(5,5)
    # print(x[None, None, :, :])
    # img_open = F.unfold(x[None, None, :, :], kernel_size=5, padding=1)
    # # out = img_open[0] - img_open[0][:, 4:5]
    # print(img_open[0] - img_open[0][:,4:5])