import torch
from typing import Sequence, Tuple, Union
import torch.nn as nn
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from src.models.networks.module import ResEncoder
import torch.nn.functional as F


class Conv3d_wd(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                                keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1,
              bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
        super(Conv3dBlock, self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1,
                                    bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes,  norm_cfg, activation_cfg,kernel_size=3, stride=1, padding=1,
                                    bias=False, weight_std=weight_std)
        self.transconv = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=1, stride=1, bias=False,
                                     weight_std=weight_std)

    def forward(self, x):
        residual = x
        out = self.resconv1(x)
        out = self.resconv2(out)
        if out.shape[1] != residual.shape[1]:
            residual = self.transconv(residual)
        out = out + residual
        return out


class SC_Net(nn.Module):
    def __init__(self,
                 in_channels: 512,
                 out_features: 2,
                 img_size: Union[Sequence[int], int],
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 pos_embed: str = "conv",
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3,
                 deep_supervision=False,
                 norm_cfg='BN', activation_cfg='ReLU', weight_std=False
                 ):
        super().__init__()
        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(2, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self._deep_supervision = deep_supervision
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.resencoder = ResEncoder(depth=10)

        # skip upsample
        self.transposeconv_skip3 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip2 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip1_0 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip1_1 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip0_0 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip0_1 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip0_2 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)


        # decoder upsample
        self.transposeconv_stage2 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)



        # decoder resnet
        self.stage3_de = ResBlock(1024, 512, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage2_de = ResBlock(512, 256, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(256, 128, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)


        # skip cnn
        self.cnn_skip2 = Conv3dBlock(512, 512, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip1_0 = Conv3dBlock(512, 512, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip1_1 = Conv3dBlock(256, 256, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip0_0 = Conv3dBlock(512, 512, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip0_1 = Conv3dBlock(256, 256, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip0_2 = Conv3dBlock(128, 128, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn0_0 = Conv3dBlock(1, 64, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn0_1 = Conv3dBlock(64, 64, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)


        # ag cnn
        self.ag_cnn1 = nn.Conv3d(512, 512, kernel_size=1)
        self.ag_cnn2 = nn.Conv3d(256, 256, kernel_size=1)
        self.ag_cnn3 = nn.Conv3d(128, 128, kernel_size=1)
        self.ag_cnn4 = nn.Conv3d(64, 64, kernel_size=1)


        self.cls_conv = nn.Conv3d(64, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.liner0 = nn.Linear(768, 512)
        self.liner1 = nn.Linear(512, out_features)


        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x):
        res_encoder_output = self.resencoder(x)
        transencoder_output, hidden_states_out = self.vit(res_encoder_output[-1])

        # classification
        cls_out = self.avgpool(self.proj_feat(transencoder_output))
        cls_out = cls_out.view(cls_out[0], -1)
        cls_out = self.liner1(self.liner0(cls_out))

        # segmentation
        skip3 = self.transposeconv_skip3(self.proj_feat(transencoder_output))
        skip2 = self.cnn_skip2(self.transposeconv_skip2(self.proj_feat(hidden_states_out[-4])))

        #ag1
        ag1_cnn1 = self.ag_cnn1(skip3)
        ag1_cnn2 = self.ag_cnn1(skip2)
        ag1_alpha1 = ag1_cnn1 + ag1_cnn2
        ag1_cnn3 = self.ag_cnn1(self.relu(ag1_alpha1))
        ag1_alpha2 = self.sigmoid(ag1_cnn3)
        ag1_out = torch.mul(skip3, ag1_alpha2)
        #######

        out1 = torch.cat([ag1_out, res_encoder_output[-1]], dim=1)
        out1 = self.stage3_de(out1)
        out1 = self.transposeconv_stage2(out1)
        skip1_0 = self.cnn_skip1_0(self.transposeconv_skip1_0(self.proj_feat(hidden_states_out[-7])))
        skip1_1 = self.cnn_skip1_1(self.transposeconv_skip1_1(skip1_0))

        #ag2
        ag2_cnn1 = self.ag_cnn2(out1)
        ag2_cnn2 = self.ag_cnn2(skip1_1)
        ag2_alpha1 = ag2_cnn1 + ag2_cnn2
        ag2_cnn3 = self.ag_cnn2(self.relu(ag2_alpha1))
        ag2_alpha2 = self.sigmoid(ag2_cnn3)
        ag2_out = torch.mul(out1, ag2_alpha2)
        ######

        out2 = torch.cat([ag2_out, res_encoder_output[-2]], dim=1)
        out2 = self.stage2_de(out2)
        out2 = self.transposeconv_stage1(out2)
        skip0_0 = self.cnn_skip0_0(self.transposeconv_skip0_0(self.proj_feat(hidden_states_out[-10])))
        skip0_1 = self.cnn_skip0_1(self.transposeconv_skip0_1(skip0_0))
        skip0_2 = self.cnn_skip0_2(self.transposeconv_skip0_2(skip0_1))

        #ag3
        ag3_cnn1 = self.ag_cnn3(out2)
        ag3_cnn2 = self.ag_cnn3(skip0_2)
        ag3_alpha1 = ag3_cnn1 + ag3_cnn2
        ag3_cnn3 = self.ag_cnn3(self.relu(ag3_alpha1))
        ag3_alpha2 = self.sigmoid(ag3_cnn3)
        ag3_out = torch.mul(out2, ag3_alpha2)
        ######

        out3 = torch.cat([ag3_out, res_encoder_output[-3]], dim=1)
        out3 = self.stage1_de(out3)
        out3 = self.transposeconv_stage0(out3)
        skip_oi = self.cnn0_1(self.cnn0_0(x))

        #ag4
        ag4_cnn1 = self.ag_cnn4(out3)
        ag4_cnn2 = self.ag_cnn4(skip_oi)
        ag4_alpha1 = ag4_cnn1 + ag4_cnn2
        ag4_cnn3 = self.ag_cnn4(self.relu(ag4_alpha1))
        ag4_alpha2 = self.sigmoid(ag4_cnn3)
        ag4_out = torch.mul(out3, ag4_alpha2)
        ######

        out4 = self.stage0_de(ag4_out)

        seg_out = self.cls_conv(out4)

        return cls_out, seg_out









