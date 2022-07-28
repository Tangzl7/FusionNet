import cv2
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
laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
avgpool = torch.tensor([[1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

def conv_operator(filename, in_channels=1):
    img = cv2.imread(filename, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    dx = F.conv2d(x, sobel_x.repeat(1, in_channels, 1, 1), stride=1, padding=1,)
    dy = F.conv2d(x, sobel_y.repeat(1, in_channels, 1, 1), stride=1, padding=1,)
    out = torch.sqrt((dx * dx + dy * dy)/2)
    print(out.shape)
    out = out.squeeze(0).numpy().transpose(1, 2, 0)

    return img, out


if __name__=="__main__":
    img_name = '../data/0001_rgb.jpg'

    img, y = conv_operator(img_name, 3)
    cv2.imwrite('edge.png', y)