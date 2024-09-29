import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import numpy as np
from timm import create_model

import fastai
from fastai.vision.all import *


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda(1).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)




def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


def conv2dTranspose_single(in_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2))


def masking(in_channels):
    return nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(in_channels, 2, 1),
        nn.AdaptiveMaxPool2d(1),
        nn.Sigmoid()
    )


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


def conv2dTranspose(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                         nn.ReLU(inplace=True),
                         nn.BatchNorm2d(num_features=out_channels))


class decoder_block(nn.Module):
    def __init__(self, in_channels=512, enc_channels=256, out_channels=128):
        super().__init__()
        self.cnn_upsamlple = conv2dTranspose(in_channels, int(in_channels // 2))
        self.double_conv = double_conv(int(in_channels // 2) + enc_channels, enc_channels)
        self.single_conv = single_conv(int(in_channels // 2) + 2 * enc_channels, out_channels)

    def forward(self, x, enc_x):
        x = self.cnn_upsamlple(x)
        cat_x = torch.cat([x, enc_x], dim=1)  # 256 + 256
        x = self.double_conv(cat_x)
        cat_x = torch.cat([x, cat_x], dim=1)
        x = self.single_conv(cat_x)
        return x




class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(out_ch * 2),
                           # GRN(out_ch*2),

                           nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1))
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer):
        hcs = [F.interpolate(c(x), scale_factor=2 ** (len(self.convs) - i), mode='bilinear')
               for i, (c, x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class UnetBlock(Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False,
                 self_attention: bool = False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
                     [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                         nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SegNext(nn.Module):
    def __init__(self, stride=1, num_classes=1, **kwargs):
        super().__init__()
        # encoder
        enc = timm.create_model("convnext_tiny", pretrained=True)

        self.econv0 = enc.stem  # 96   ,256X256
        self.econv1 = enc.stages[0]  # 96
        self.econv2 = enc.stages[1]  # 192
        self.econv3 = enc.stages[2]  # 384
        self.econv4 = enc.stages[3]  # 768

        # aspp with customized dilatations
        self.aspp = ASPP(inplanes=768, mid_c=192, out_c=384, dilations=[stride * 1, stride * 2, stride * 3, stride * 4])
        self.drop_aspp = nn.Dropout2d(0.25)
        # decoder
        self.dec4 = UnetBlock(384, 384, 192)
        self.dec3 = UnetBlock(192, 192, 96)
        self.dec2 = UnetBlock(96, 96, 96)

        self.fpn = FPN([384, 192, 96, 96], [16] * 4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(96 + 16 * 4, num_classes * 4, ks=1, norm_type=None, act_cls=None)
        self.segment_head = nn.Conv2d(num_classes * 4, num_classes, 1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        #x: [bs,channels,1024,1024]
        conv0 = self.econv0(x)  #  [bs,96,256,256]
        conv1 = self.econv1(conv0)  # [bs,96,256,256]
        conv2 = self.econv2(conv1)  # [bs,192,128,128]
        conv3 = self.econv3(conv2)  # [bs,384,64,64]
        conv4 = self.econv4(conv3)  # [bs,768,32,32]
        conv5 = self.aspp(conv4) # [bs,384,32,32]
        dec4 = self.dec4(self.drop_aspp(conv5), conv3) # [bs,192,64,64]
        dec3 = self.dec3(dec4, conv2) # [bs,96,128,128]
        dec2 = self.dec2(dec3, conv1) # [bs,96,256,256]
        dec1 = self.dec2(dec2, self.upsample(conv0)) #[bs,96,512,512]
        x = self.fpn([conv5, dec4, dec3, dec2], dec1) #[bs,160,512,512]
        x = self.final_conv(self.drop(x)) ##[bs,num_classes*4,512,512]
        x = self.segment_head(x) #[bs,num_classes,512,512]
        x = self.upsample(x) #[bs,num_classes,1024,1024]
        return x




# device  = torch.device('cuda:'+ str(1))
# model   = SegNext(num_classes=2).cuda(device=device)
# print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
# x1 = torch.rand([2,3,128,128])
# #x2 = torch.rand([2,3,128,128])
#
#
# x1 = x1.cuda(device)
# #x1, = type('torch.cuda.FloatTensor')
#
#
# with torch.set_grad_enabled(False):
#     with torch.cuda.amp.autocast():
#         #outputs1 = model(x1,x2)
#         outputs1 = model(x1)
#         print(outputs1)
#
#
