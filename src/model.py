import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import pdb
import numpy as np
from PIL import Image


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
        
        modules.append(CABlock(n_feat, kernel_size, reduction, bias, act))
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
