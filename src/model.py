import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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
    def __init__(self, conv_cnt, n_feat, kernel_size, reduction, act, bias, scale_unet_feats, scale_factor):
        super(Encoder, self).__init__()
        modules_body = []
        for i in range(conv_cnt):
            modules_body.append(nn.Conv2d(n_feat, scale_unet_feats * (2 ** i), kernel_size, padding='same' , bias=bias))
            modules_body.append(act)
            modules_body.append(Sample(scale_factor))
            n_feat = scale_unet_feats * (2 ** i)
        self.encode = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.encode(x)
        return out

class Decoder(nn.Module):
    def __init__(self, conv_cnt, n_feat, kernel_size, reduction, act, bias, scale_unet_feats, scale_factor):
        super(Decoder, self).__init__()
        modules_body = []
        for i in range(conv_cnt-1, -1, -1):
            if i == 0:
                modules_body.append(nn.Conv2d(scale_unet_feats * (2 ** i), n_feat, kernel_size, padding='same' , bias=bias))
            else:
                modules_body.append(nn.Conv2d(scale_unet_feats * (2 ** i), scale_unet_feats * (2 ** (i-1)), kernel_size, padding='same' , bias=bias))
            modules_body.append(act)
            modules_body.append(Sample(scale_factor))
        self.decode = nn.Sequential(*modules_body)
    
    def forward(self, x):
        out = self.decode(x)
        return out


class FusionNet(nn.Module):
    def __init__(self, vis_c=3, nir_c=1, out_c=3, feat_conv_cnt=3, feat_scale_factor=16, kernel_size=3, reduction=4, bias=False):
        super(FusionNet, self).__init__()
        act = nn.PReLU()
        self.vis_feat_extractor = FeatExtractor(feat_conv_cnt, feat_scale_factor, vis_c, kernel_size, reduction, bias, act)
        self.nir_feat_extractor = FeatExtractor(feat_conv_cnt, feat_scale_factor, nir_c, kernel_size, reduction, bias, act)

    def forward(self, vis, nir):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        feat_1 = self.vis_feat_extractor(vis)
        feat_2 = self.nir_feat_extractor(nir)
        return feat_1, feat_2


if __name__ == '__main__':
    fusion_net = FusionNet()
    rgb, nir = Image.open('../data/0001_rgb.jpg'), Image.open('../data/0001_nir.jpg')
    rgb, nir = TF.to_tensor(rgb), TF.to_tensor(nir)
    nir = nir[0, :, :].reshape(1, 2160, 3840)
    print(fusion_net)
    out1, out2 = fusion_net(rgb, nir)
    print(out2.shape)
    pass
