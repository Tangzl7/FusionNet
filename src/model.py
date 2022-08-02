from audioop import bias
from bisect import bisect
from turtle import forward
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import pdb
import numpy as np
from PIL import Image

class NirExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, feats=32, kernel_size=3, bias=True):
        super(NirExtractor, self).__init__()
        self.act = nn.PReLU()
        self.modules = []
        self.modules.append(nn.Conv2d(in_channels, feats, kernel_size, padding='same', bias=True))
        self.modules.append(self.act)
        self.modules.append(nn.Conv2d(feats, out_channels, kernel_size, padding='same', bias=True))
        self.modules.append(self.act)
        self.extractor = nn.Sequential(*self.modules)

    def forward(self, x):
        out = self.extractor(x)
        return out


class DenoisyNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, feats=32, kernel_size=3, bias=True):
        super(DenoisyNet, self).__init__()
        self.kernels_0 = torch.tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 15, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]], 
                        dtype=torch.float, requires_grad=True).view(1, 1, 5, 5)
        self.kernels_1 = torch.tensor([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]], 
                        dtype=torch.float, requires_grad=True).view(1, 1, 3, 3)
        self.kernels_2 = torch.tensor([[1]], 
                        dtype=torch.float, requires_grad=True).view(1, 1, 1, 1)
        self.kernels_3 = torch.tensor([[1/9 for j in range(3)] for i in range(3)], 
                        dtype=torch.float, requires_grad=True).view(1, 1, 3, 3)
        self.kernels_4 = torch.tensor([[1/25 for j in range(5)] for i in range(5)], 
                        dtype=torch.float, requires_grad=True).view(1, 1, 5, 5)
        self.kernels_5 = torch.tensor([[1/49 for j in range(7)] for i in range(7)], 
                        dtype=torch.float, requires_grad=True).view(1, 1, 7, 7)
        self.kernels_6 = torch.tensor([[1/81 for j in range(9)] for i in range(9)], 
                        dtype=torch.float, requires_grad=True).view(1, 1, 9, 9)
        if torch.cuda.is_available():
            self.kernels_0 = self.kernels_0.cuda()
            self.kernels_1 = self.kernels_1.cuda()
            self.kernels_2 = self.kernels_2.cuda()
            self.kernels_3 = self.kernels_3.cuda()
            self.kernels_4 = self.kernels_4.cuda()
            self.kernels_5 = self.kernels_5.cuda()
            self.kernels_6 = self.kernels_6.cuda()

        self.nir_extractor = NirExtractor()

        self.act = nn.PReLU()
        self.encode_conv1 = nn.Conv2d(in_channels, feats, kernel_size, padding='same', bias=bias)
        self.encode_conv2 = nn.Conv2d(feats, feats, kernel_size, padding='same', bias=bias)
        self.encode_conv3 = nn.Conv2d(feats, feats, kernel_size, padding='same', bias=bias)
        self.decode_conv1 = nn.Conv2d(feats*2, feats, kernel_size, padding='same', bias=bias)
        self.decode_conv2 = nn.Conv2d(feats*2, feats, kernel_size, padding='same', bias=bias)
        self.decode_conv3 = nn.Conv2d(feats+in_channels, out_channels, kernel_size, padding='same', bias=bias)
    
    def forward(self, x, y):
        nir_feats = self.nir_extractor(y)

        x1 = self.act(self.encode_conv1(x)) + nir_feats
        x2 = self.act(self.encode_conv2(x1))
        x3 = self.act(self.encode_conv3(x2))
        x4 = self.act(self.decode_conv1(torch.cat([x2, x3], 1)))
        x5 = self.act(self.decode_conv2(torch.cat([x1, x4], 1)))
        x6 = 6 * torch.sigmoid(self.decode_conv3(torch.cat([x, x5], 1)))
        denoisy_map = torch.round(x6)

        out_map_1_0 = F.conv2d(x, self.kernels_0, padding=2)
        out_map_1_1 = F.conv2d(x, self.kernels_1, padding=1)
        out_map_1_2 = F.conv2d(x, self.kernels_2, padding=0)
        out_map_1_3 = F.conv2d(x, self.kernels_3, padding=1)
        out_map_1_4 = F.conv2d(x, self.kernels_4, padding=2)
        out_map_1_5 = F.conv2d(x, self.kernels_5, padding=3)
        out_map_1_6 = F.conv2d(x, self.kernels_6, padding=4)
        denoisy_1_0 = torch.where(denoisy_map[:, 0, :, :] == 0, out_map_1_0, x)
        denoisy_1_1 = torch.where(denoisy_map[:, 0, :, :] == 1, out_map_1_1, denoisy_1_0)
        denoisy_1_2 = torch.where(denoisy_map[:, 0, :, :] == 2, out_map_1_2, denoisy_1_1)
        denoisy_1_3 = torch.where(denoisy_map[:, 0, :, :] == 3, out_map_1_3, denoisy_1_2)
        denoisy_1_4 = torch.where(denoisy_map[:, 0, :, :] == 4, out_map_1_4, denoisy_1_3)
        denoisy_1_5 = torch.where(denoisy_map[:, 0, :, :] == 5, out_map_1_5, denoisy_1_4)
        denoisy_1 = torch.where(denoisy_map[:, 0, :, :] == 6, out_map_1_6, denoisy_1_5)

        out_map_2_0 = F.conv2d(denoisy_1, self.kernels_0, padding=2)
        out_map_2_1 = F.conv2d(denoisy_1, self.kernels_1, padding=1)
        out_map_2_2 = F.conv2d(denoisy_1, self.kernels_2, padding=0)
        out_map_2_3 = F.conv2d(denoisy_1, self.kernels_3, padding=1)
        out_map_2_4 = F.conv2d(denoisy_1, self.kernels_4, padding=2)
        out_map_2_5 = F.conv2d(denoisy_1, self.kernels_5, padding=3)
        out_map_2_6 = F.conv2d(denoisy_1, self.kernels_6, padding=4)
        denoisy_2_0 = torch.where(denoisy_map[:, 1, :, :] == 0, out_map_2_0, denoisy_1)
        denoisy_2_1 = torch.where(denoisy_map[:, 1, :, :] == 1, out_map_2_1, denoisy_2_0)
        denoisy_2_2 = torch.where(denoisy_map[:, 1, :, :] == 2, out_map_2_2, denoisy_2_1)
        denoisy_2_3 = torch.where(denoisy_map[:, 1, :, :] == 3, out_map_2_3, denoisy_2_2)
        denoisy_2_4 = torch.where(denoisy_map[:, 1, :, :] == 4, out_map_2_4, denoisy_2_3)
        denoisy_2_5 = torch.where(denoisy_map[:, 1, :, :] == 5, out_map_2_5, denoisy_2_4)
        denoisy_2 = torch.where(denoisy_map[:, 1, :, :] == 6, out_map_2_6, denoisy_2_5)

        out_map_3_0 = F.conv2d(denoisy_2, self.kernels_0, padding=2)
        out_map_3_1 = F.conv2d(denoisy_2, self.kernels_1, padding=1)
        out_map_3_2 = F.conv2d(denoisy_2, self.kernels_2, padding=0)
        out_map_3_3 = F.conv2d(denoisy_2, self.kernels_3, padding=1)
        out_map_3_4 = F.conv2d(denoisy_2, self.kernels_4, padding=2)
        out_map_3_5 = F.conv2d(denoisy_2, self.kernels_5, padding=3)
        out_map_3_6 = F.conv2d(denoisy_2, self.kernels_6, padding=4)
        denoisy_3_0 = torch.where(denoisy_map[:, 2, :, :] == 0, out_map_3_0, denoisy_2)
        denoisy_3_1 = torch.where(denoisy_map[:, 2, :, :] == 1, out_map_3_1, denoisy_3_0)
        denoisy_3_2 = torch.where(denoisy_map[:, 2, :, :] == 2, out_map_3_2, denoisy_3_1)
        denoisy_3_3 = torch.where(denoisy_map[:, 2, :, :] == 3, out_map_3_3, denoisy_3_2)
        denoisy_3_4 = torch.where(denoisy_map[:, 2, :, :] == 4, out_map_3_4, denoisy_3_3)
        denoisy_3_5 = torch.where(denoisy_map[:, 2, :, :] == 5, out_map_3_5, denoisy_3_4)
        denoisy_3 = torch.where(denoisy_map[:, 2, :, :] == 6, out_map_3_6, denoisy_3_5)

        out_map_4_0 = F.conv2d(denoisy_3, self.kernels_0, padding=2)
        out_map_4_1 = F.conv2d(denoisy_3, self.kernels_1, padding=1)
        out_map_4_2 = F.conv2d(denoisy_3, self.kernels_2, padding=0)
        out_map_4_3 = F.conv2d(denoisy_3, self.kernels_3, padding=1)
        out_map_4_4 = F.conv2d(denoisy_3, self.kernels_4, padding=2)
        out_map_4_5 = F.conv2d(denoisy_3, self.kernels_5, padding=3)
        out_map_4_6 = F.conv2d(denoisy_3, self.kernels_6, padding=4)
        denoisy_4_0 = torch.where(denoisy_map[:, 3, :, :] == 0, out_map_4_0, denoisy_3)
        denoisy_4_1 = torch.where(denoisy_map[:, 3, :, :] == 1, out_map_4_1, denoisy_4_0)
        denoisy_4_2 = torch.where(denoisy_map[:, 3, :, :] == 2, out_map_4_2, denoisy_4_1)
        denoisy_4_3 = torch.where(denoisy_map[:, 3, :, :] == 3, out_map_4_3, denoisy_4_2)
        denoisy_4_4 = torch.where(denoisy_map[:, 3, :, :] == 4, out_map_4_4, denoisy_4_3)
        denoisy_4_5 = torch.where(denoisy_map[:, 3, :, :] == 5, out_map_4_5, denoisy_4_4)
        denoisy_4 = torch.where(denoisy_map[:, 3, :, :] == 6, out_map_4_6, denoisy_4_5)

        out_map_5_0 = F.conv2d(denoisy_4, self.kernels_0, padding=2)
        out_map_5_1 = F.conv2d(denoisy_4, self.kernels_1, padding=1)
        out_map_5_2 = F.conv2d(denoisy_4, self.kernels_2, padding=0)
        out_map_5_3 = F.conv2d(denoisy_4, self.kernels_3, padding=1)
        out_map_5_4 = F.conv2d(denoisy_4, self.kernels_4, padding=2)
        out_map_5_5 = F.conv2d(denoisy_4, self.kernels_5, padding=3)
        out_map_5_6 = F.conv2d(denoisy_4, self.kernels_6, padding=4)
        denoisy_5_0 = torch.where(denoisy_map[:, 4, :, :] == 0, out_map_5_0, denoisy_4)
        denoisy_5_1 = torch.where(denoisy_map[:, 4, :, :] == 1, out_map_5_1, denoisy_5_0)
        denoisy_5_2 = torch.where(denoisy_map[:, 4, :, :] == 2, out_map_5_2, denoisy_5_1)
        denoisy_5_3 = torch.where(denoisy_map[:, 4, :, :] == 3, out_map_5_3, denoisy_5_2)
        denoisy_5_4 = torch.where(denoisy_map[:, 4, :, :] == 4, out_map_5_4, denoisy_5_3)
        denoisy_5_5 = torch.where(denoisy_map[:, 4, :, :] == 5, out_map_5_5, denoisy_5_4)
        denoisy_5 = torch.where(denoisy_map[:, 4, :, :] == 6, out_map_5_6, denoisy_5_5)

        out_map_6_0 = F.conv2d(denoisy_5, self.kernels_0, padding=2)
        out_map_6_1 = F.conv2d(denoisy_5, self.kernels_1, padding=1)
        out_map_6_2 = F.conv2d(denoisy_5, self.kernels_2, padding=0)
        out_map_6_3 = F.conv2d(denoisy_5, self.kernels_3, padding=1)
        out_map_6_4 = F.conv2d(denoisy_5, self.kernels_4, padding=2)
        out_map_6_5 = F.conv2d(denoisy_5, self.kernels_5, padding=3)
        out_map_6_6 = F.conv2d(denoisy_5, self.kernels_6, padding=4)
        denoisy_6_0 = torch.where(denoisy_map[:, 5, :, :] == 0, out_map_6_0, denoisy_5)
        denoisy_6_1 = torch.where(denoisy_map[:, 5, :, :] == 1, out_map_6_1, denoisy_6_0)
        denoisy_6_2 = torch.where(denoisy_map[:, 5, :, :] == 2, out_map_6_2, denoisy_6_1)
        denoisy_6_3 = torch.where(denoisy_map[:, 5, :, :] == 3, out_map_6_3, denoisy_6_2)
        denoisy_6_4 = torch.where(denoisy_map[:, 5, :, :] == 4, out_map_6_4, denoisy_6_3)
        denoisy_6_5 = torch.where(denoisy_map[:, 5, :, :] == 5, out_map_6_5, denoisy_6_4)
        denoisy_6 = torch.where(denoisy_map[:, 5, :, :] == 6, out_map_6_6, denoisy_6_5)

        out_map_7_0 = F.conv2d(denoisy_6, self.kernels_0, padding=2)
        out_map_7_1 = F.conv2d(denoisy_6, self.kernels_1, padding=1)
        out_map_7_2 = F.conv2d(denoisy_6, self.kernels_2, padding=0)
        out_map_7_3 = F.conv2d(denoisy_6, self.kernels_3, padding=1)
        out_map_7_4 = F.conv2d(denoisy_6, self.kernels_4, padding=2)
        out_map_7_5 = F.conv2d(denoisy_6, self.kernels_5, padding=3)
        out_map_7_6 = F.conv2d(denoisy_6, self.kernels_6, padding=4)
        denoisy_7_0 = torch.where(denoisy_map[:, 6, :, :] == 0, out_map_7_0, denoisy_6)
        denoisy_7_1 = torch.where(denoisy_map[:, 6, :, :] == 1, out_map_7_1, denoisy_7_0)
        denoisy_7_2 = torch.where(denoisy_map[:, 6, :, :] == 2, out_map_7_2, denoisy_7_1)
        denoisy_7_3 = torch.where(denoisy_map[:, 6, :, :] == 3, out_map_7_3, denoisy_7_2)
        denoisy_7_4 = torch.where(denoisy_map[:, 6, :, :] == 4, out_map_7_4, denoisy_7_3)
        denoisy_7_5 = torch.where(denoisy_map[:, 6, :, :] == 5, out_map_7_5, denoisy_7_4)
        out = torch.where(denoisy_map[:, 6, :, :] == 6, out_map_7_6, denoisy_7_5)

        return out

if __name__ == '__main__':
    denoisy_net = DenoisyNet().cuda()
    denoisy_net.load_state_dict(torch.load('./snapshots/denoisy.pth'))
    rgb, nir = cv2.imread('../data/rgb/0001_rgb.jpg'), cv2.imread('../data/nir/0001_nir.jpg', 0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    lab = lab / 255.
    nir = nir / 255.
    l_channel = lab[:, :, 0]
    l_channel, nir = TF.to_tensor(l_channel), TF.to_tensor(nir)
    l_channel, nir = torch.unsqueeze(l_channel, 0), torch.unsqueeze(nir, 0)
    # pdb.set_trace()
    print(denoisy_net)
    print(l_channel.shape)
    out = denoisy_net(l_channel.float().cuda(), nir.float().cuda()) * 255.
    out = torch.squeeze(out, 0)
    out = out.cpu().detach().numpy()
    out = np.transpose(out, (1, 2, 0))
    cv2.imwrite('tmp.png', np.uint8(out))
'''
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv_du(out)
        return x * out


class CABlock(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CABlock, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding='same', bias=bias))
        modules_body.append(act)
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding='same', bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.body(x)
        out = self.CA(out)
        out += x
        return out

class FeatExtractor(nn.Module):
    def __init__(self, conv_cnt, scale_feats, n_feat, kernel_size, reduction, bias, act):
        super(FeatExtractor, self).__init__()
        modules = []
        for i in range(conv_cnt):
            modules.append(nn.Conv2d(n_feat, scale_feats * (2 ** i), kernel_size, padding='same', bias=bias))
            modules.append(act)
            n_feat = 16 * (2 ** i)
        
        # modules.append(CABlock(n_feat, kernel_size, reduction, bias, act))
        self.extractor = nn.Sequential(*modules)

    def forward(self, x):
        out = self.extractor(x)
        return out


class Sample(nn.Module):
    def __init__(self, scale_factor):
        super(Sample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))

    def forward(self, x):
        out = self.down(x)
        return out

class Encoder(nn.Module):
    def __init__(self, conv_cnt, n_feat, kernel_size, act, bias, scale_unet_feats, scale_factor):
        super(Encoder, self).__init__()
        self.enc1 = nn.Conv2d(n_feat, scale_unet_feats, kernel_size, padding='same' , bias=bias)
        self.enc2 = nn.Conv2d(scale_unet_feats, scale_unet_feats * 2, kernel_size, padding='same' , bias=bias)
        self.enc3 = nn.Conv2d(scale_unet_feats * 2, scale_unet_feats * 4, kernel_size, padding='same' , bias=bias)
        self.act = act
        self.down = Sample(scale_factor)

    def forward(self, x):
        enc1 = self.enc1(x)
        x = self.act(enc1)
        x = self.down(x)
        enc2 = self.enc2(x)
        x = self.act(enc2)
        x = self.down(x)
        enc3 = self.enc3(x)
        return enc1, enc2, enc3

class Decoder(nn.Module):
    def __init__(self, conv_cnt, n_feat, kernel_size, act, bias, scale_unet_feats, scale_factor):
        super(Decoder, self).__init__()
        modules_body = []
        self.dec1 = nn.Conv2d(scale_unet_feats * 4, scale_unet_feats * 2, kernel_size, padding='same' , bias=bias)
        self.dec2 = nn.Conv2d(scale_unet_feats * 2, scale_unet_feats, kernel_size, padding='same' , bias=bias)
        self.dec3 = nn.Conv2d(scale_unet_feats, n_feat, kernel_size, padding='same' , bias=bias)
        self.act = act
        self.up = Sample(scale_factor)
    
    def forward(self, x):
        enc1, enc2, enc3 = x
        dec1 = self.dec1(enc3)
        t = enc2 + self.up(dec1)
        dec2 = self.dec2(t)
        t = enc1 + self.up(dec2)
        out = self.dec3(t)
        return out


class FNet(nn.Module):
    def __init__(self, conv_cnt, scale_feats, n_feat, kernel_size, reduction, bias, act):
        super(FNet, self).__init__()
        modules = []
        modules.append(CABlock(scale_feats * (2 ** (conv_cnt-1)), kernel_size, reduction, bias, act))
        for i in range(conv_cnt-1, -1, -1):
            if i == 0:
                modules.append(nn.Conv2d(scale_feats * (2 ** i), n_feat, kernel_size, padding='same' , bias=bias))
            else:
                modules.append(nn.Conv2d(scale_feats * (2 ** i), scale_feats * (2 ** (i-1)), kernel_size, padding='same', bias=bias))
            modules.append(act)
        
        self.fusion = nn.Sequential(*modules)
    
    def forward(self, x, y):
        out = x + y
        out = self.fusion(out)
        return out

class FusionNet(nn.Module):
    def __init__(self, vis_c=3, nir_c=1, out_c=3, feat_conv_cnt=3, feat_scale_factor=16, coder_conv_cnt=3, scale_unet_feats=16, kernel_size=3, reduction=4, bias=False):
        super(FusionNet, self).__init__()
        act = nn.PReLU()
        self.act = nn.PReLU()
        self.vis_feat_extractor = FeatExtractor(feat_conv_cnt, feat_scale_factor, vis_c, kernel_size, reduction, bias, act)
        self.nir_feat_extractor = FeatExtractor(feat_conv_cnt, feat_scale_factor, nir_c, kernel_size, reduction, bias, act)
        self.encoder = Encoder(coder_conv_cnt, vis_c, kernel_size, act, bias, scale_unet_feats, scale_factor=0.5)
        self.decoder = Decoder(coder_conv_cnt, out_c, kernel_size, act, bias, scale_unet_feats, scale_factor=2)
        self.decoder_tail = nn.Conv2d(out_c, out_c, kernel_size, padding='same')
        self.fnet = FNet(feat_conv_cnt, feat_scale_factor, out_c, kernel_size, reduction, bias, act)
        self.fnet_tail = nn.Conv2d(out_c, out_c, kernel_size, padding='same')

    def forward(self, vis, nir):
        # pdb.set_trace()
        feat_1 = self.vis_feat_extractor(vis)
        feat_2 = self.nir_feat_extractor(nir)
        fusion = self.fnet(feat_1, feat_2)

        coder_out = self.encoder(vis)
        coder_out = self.decoder(coder_out)
        coder_out = self.decoder_tail(coder_out)
        coder_out = self.act(coder_out)

        out = fusion + coder_out
        out = self.fnet_tail(out)
        return out


if __name__ == '__main__':
    fusion_net = FusionNet()
    fusion_net.load_state_dict(torch.load('../checkpoints/model_latest.pth'))
    rgb, nir = cv2.imread('../data/rgb/0002_rgb.jpg'), cv2.imread('../data/nir/0002_nir.jpg', 0)
    rgb = cv2.resize(rgb, (1920, 1080)) / 255.
    nir = cv2.resize(nir, (1920, 1080)) / 255.
    rgb, nir = TF.to_tensor(rgb), TF.to_tensor(nir)
    rgb, nir = torch.unsqueeze(rgb, 0), torch.unsqueeze(nir, 0)
    # pdb.set_trace()
    print(fusion_net)
    out = fusion_net(rgb.float(), nir.float()) * 255.
    out = torch.squeeze(out, 0)
    out = out.detach().numpy()
    out = np.transpose(out, (1, 2, 0))
    cv2.imwrite('./out.png', np.uint8(out))
'''