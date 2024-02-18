# -*- coding: utf-8 -*-
# Time    : 2023/12/27 16:21
# Author  : fanc
# File    : net.py
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
        self.num_layers = 15
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(1, spatial_dims)
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

        # skip upsample
        self.transposeconv_skip4 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip3 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip2_1 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip2_2 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip1_1 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip1_2 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip1_3 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip0_1 = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip0_2 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip0_3 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_skip0_4 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)



        # decoder upsample
        self.transposeconv_stage3 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)


        # decoder resnet
        self.stage4_de = ResBlock(1024, 512, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage3_de = ResBlock(512, 256, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage2_de = ResBlock(256, 128, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(128, 64, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)

        # resencoder output upsample
        self.transposeconv_resout_up0 = nn.ConvTranspose3d(512, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_resout_up1 = nn.ConvTranspose3d(256, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_resout_up2 = nn.ConvTranspose3d(128, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_resout_up3 = nn.ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)

        # skip cnn
        self.cnn_skip3 = Conv3dBlock(512, 512, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip2_1 = Conv3dBlock(512, 512, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip2_2 = Conv3dBlock(256, 256, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip1_1 = Conv3dBlock(512, 512, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip1_2 = Conv3dBlock(256, 256, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip1_3 = Conv3dBlock(128, 128, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip0_1 = Conv3dBlock(512, 512, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip0_2 = Conv3dBlock(256, 256, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip0_3 = Conv3dBlock(128, 128, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cnn_skip0_4 = Conv3dBlock(64, 64, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cbr1 = Conv3dBlock(1, 64, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)
        self.cbr2 = Conv3dBlock(64, 64, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, weight_std=weight_std)

        # ag cnn
        self.ag_cnn1 = nn.Conv3d(512, 512, kernel_size=1)
        self.ag_cnn2 = nn.Conv3d(256, 256, kernel_size=1)
        self.ag_cnn3 = nn.Conv3d(128, 128, kernel_size=1)
        self.ag_cnn4 = nn.Conv3d(64, 64, kernel_size=1)
        self.ag_cnn5 = nn.Conv3d(64, 64, kernel_size=1)

        self.cls_conv = nn.Conv3d(64, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def seg_forword(self, res_encoder_output):
        transencoder_output, hidden_states_out = self.vit(res_encoder_output[-1])
        # self.transencoder_output = self.proj_feat(transencoder_output)
        skip4 = self.transposeconv_skip4(self.proj_feat(transencoder_output))
        skip3 = self.cnn_skip3(self.transposeconv_skip3(self.proj_feat(hidden_states_out[-4])))

        #ag1
        ag1_cnn1 = self.ag_cnn1(skip4)
        ag1_cnn2 = self.ag_cnn1(skip3)
        ag1_alpha1 = ag1_cnn1 + ag1_cnn2
        ag1_cnn3 = self.ag_cnn1(self.relu(ag1_alpha1))
        ag1_alpha2 = self.sigmoid(ag1_cnn3)
        ag1_out = torch.mul(skip4, ag1_alpha2)
        #######

        resout_up0 = self.transposeconv_resout_up0(res_encoder_output[-1])
        out1 = torch.cat([ag1_out, resout_up0], dim=1)
        out1 = self.stage4_de(out1)
        out1 = self.transposeconv_stage3(out1)
        skip2_1 = self.cnn_skip2_1(self.transposeconv_skip2_1(self.proj_feat(hidden_states_out[-7])))
        skip2_2 = self.cnn_skip2_2(self.transposeconv_skip2_2(skip2_1))

        #ag2
        ag2_cnn1 = self.ag_cnn2(out1)
        ag2_cnn2 = self.ag_cnn2(skip2_2)
        ag2_alpha1 = ag2_cnn1 + ag2_cnn2
        ag2_cnn3 = self.ag_cnn2(self.relu(ag2_alpha1))
        ag2_alpha2 = self.sigmoid(ag2_cnn3)
        ag2_out = torch.mul(out1, ag2_alpha2)
        ######

        resout_up1 = self.transposeconv_resout_up1(res_encoder_output[-2])
        out2 = torch.cat([ag2_out, resout_up1], dim=1)
        out2 = self.stage3_de(out2)
        out2 = self.transposeconv_stage2(out2)
        skip1_1 = self.cnn_skip1_1(self.transposeconv_skip1_1(self.proj_feat(hidden_states_out[-10])))
        skip1_2 = self.cnn_skip1_2(self.transposeconv_skip1_2(skip1_1))
        skip1_3 = self.cnn_skip1_3(self.transposeconv_skip1_3(skip1_2))

        #ag3
        ag3_cnn1 = self.ag_cnn3(out2)
        ag3_cnn2 = self.ag_cnn3(skip1_3)
        ag3_alpha1 = ag3_cnn1 + ag3_cnn2
        ag3_cnn3 = self.ag_cnn3(self.relu(ag3_alpha1))
        ag3_alpha2 = self.sigmoid(ag3_cnn3)
        ag3_out = torch.mul(out2, ag3_alpha2)
        ######

        resout_up2 = self.transposeconv_resout_up2(res_encoder_output[-3])
        out3 = torch.cat([ag3_out, resout_up2], dim=1)
        out3 = self.stage2_de(out3)
        out3 = self.transposeconv_stage1(out3)
        skip0_1 = self.cnn_skip0_1(self.transposeconv_skip0_1(self.proj_feat(hidden_states_out[-13])))
        skip0_2 = self.cnn_skip0_2(self.transposeconv_skip0_2(skip0_1))
        skip0_3 = self.cnn_skip0_3(self.transposeconv_skip0_3(skip0_2))
        skip0_4 = self.cnn_skip0_4(self.transposeconv_skip0_4(skip0_3))

        #ag4
        ag4_cnn1 = self.ag_cnn4(out3)
        ag4_cnn2 = self.ag_cnn4(skip0_4)
        ag4_alpha1 = ag4_cnn1 + ag4_cnn2
        ag4_cnn3 = self.ag_cnn4(self.relu(ag4_alpha1))
        ag4_alpha2 = self.sigmoid(ag4_cnn3)
        ag4_out = torch.mul(out3, ag4_alpha2)
        ######

        resout_up3 = self.transposeconv_resout_up3(res_encoder_output[-4])
        out4 = torch.cat([ag4_out, resout_up3], dim=1)
        out4 = self.stage1_de(out4)
        out4 = self.transposeconv_stage0(out4)
        cbr_skip = self.cbr2(self.cbr1(res_encoder_output[-5]))

        #ag5
        ag5_cnn1 = self.ag_cnn5(out4)
        ag5_cnn2 = self.ag_cnn5(cbr_skip)
        ag5_alpha1 = ag5_cnn1 + ag5_cnn2
        ag5_cnn3 = self.ag_cnn5(self.relu(ag5_alpha1))
        ag5_alpha2 = self.sigmoid(ag5_cnn3)
        ag5_out = torch.mul(out4, ag5_alpha2)
        ######

        out5 = self.stage0_de(ag5_out)
        out_final = self.sigmoid(self.cls_conv(out5))

    def cla_forward(self, x, y, z):
        # encode
        feature_re = x
        feature_tf = self.transconv(y)
        # feature_zoom = self.zoom_encode(y)[-2]
        feature_mask = self.mask_encode(z)[-1]
        features = torch.cat([feature_re, feature_tf, feature_mask], dim=1)
        features = self.conv1(features)
        features = self.bn1(features)
        features = self.relu(features)
        # decode
        # batch_size, channels, depth, height, width  = feature_oi.size()
        # proj_query = self.query_conv(feature_mask).view(batch_size, -1, width * height * depth).permute(0, 2, 1)  # B X CX(N)
        # proj_key = self.key_conv(feature_zoom).view(batch_size, -1, width * height * depth)  # B X C x (*W*H)
        # energy = torch.bmm(proj_query, proj_key)  # transpose check
        # attention = self.softmax(energy)  # BX (N) X (N)
        # proj_value = self.value_conv(feature_oi).view(batch_size, -1, width * height * depth)  # B X C X N
        # att_x = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # att_x = att_x.view(batch_size, channels, depth, height, width)
        # att_x = self.gamma * att_x + feature_oi
        # print('gama\n', self.gamma)
        # att_x = att_x + feature_oi
        # self.finalconv = att_x.clone()
        # classification head
        res = self.avgpool(features)
        res = res.view(res.size(0), -1)
        res = self.fc(res)

        return res
    def forward(self, res_encoder_output):
        transencoder_output, hidden_states_out = self.vit(res_encoder_output[-1])
        # self.transencoder_output = self.proj_feat(transencoder_output)
        skip4 = self.transposeconv_skip4(self.proj_feat(transencoder_output))
        skip3 = self.cnn_skip3(self.transposeconv_skip3(self.proj_feat(hidden_states_out[-4])))

        #ag1
        ag1_cnn1 = self.ag_cnn1(skip4)
        ag1_cnn2 = self.ag_cnn1(skip3)
        ag1_alpha1 = ag1_cnn1 + ag1_cnn2
        ag1_cnn3 = self.ag_cnn1(self.relu(ag1_alpha1))
        ag1_alpha2 = self.sigmoid(ag1_cnn3)
        ag1_out = torch.mul(skip4, ag1_alpha2)
        #######

        resout_up0 = self.transposeconv_resout_up0(res_encoder_output[-1])
        out1 = torch.cat([ag1_out, resout_up0], dim=1)
        out1 = self.stage4_de(out1)
        out1 = self.transposeconv_stage3(out1)
        skip2_1 = self.cnn_skip2_1(self.transposeconv_skip2_1(self.proj_feat(hidden_states_out[-7])))
        skip2_2 = self.cnn_skip2_2(self.transposeconv_skip2_2(skip2_1))

        #ag2
        ag2_cnn1 = self.ag_cnn2(out1)
        ag2_cnn2 = self.ag_cnn2(skip2_2)
        ag2_alpha1 = ag2_cnn1 + ag2_cnn2
        ag2_cnn3 = self.ag_cnn2(self.relu(ag2_alpha1))
        ag2_alpha2 = self.sigmoid(ag2_cnn3)
        ag2_out = torch.mul(out1, ag2_alpha2)
        ######

        resout_up1 = self.transposeconv_resout_up1(res_encoder_output[-2])
        out2 = torch.cat([ag2_out, resout_up1], dim=1)
        out2 = self.stage3_de(out2)
        out2 = self.transposeconv_stage2(out2)
        skip1_1 = self.cnn_skip1_1(self.transposeconv_skip1_1(self.proj_feat(hidden_states_out[-10])))
        skip1_2 = self.cnn_skip1_2(self.transposeconv_skip1_2(skip1_1))
        skip1_3 = self.cnn_skip1_3(self.transposeconv_skip1_3(skip1_2))

        #ag3
        ag3_cnn1 = self.ag_cnn3(out2)
        ag3_cnn2 = self.ag_cnn3(skip1_3)
        ag3_alpha1 = ag3_cnn1 + ag3_cnn2
        ag3_cnn3 = self.ag_cnn3(self.relu(ag3_alpha1))
        ag3_alpha2 = self.sigmoid(ag3_cnn3)
        ag3_out = torch.mul(out2, ag3_alpha2)
        ######

        resout_up2 = self.transposeconv_resout_up2(res_encoder_output[-3])
        out3 = torch.cat([ag3_out, resout_up2], dim=1)
        out3 = self.stage2_de(out3)
        out3 = self.transposeconv_stage1(out3)
        skip0_1 = self.cnn_skip0_1(self.transposeconv_skip0_1(self.proj_feat(hidden_states_out[-13])))
        skip0_2 = self.cnn_skip0_2(self.transposeconv_skip0_2(skip0_1))
        skip0_3 = self.cnn_skip0_3(self.transposeconv_skip0_3(skip0_2))
        skip0_4 = self.cnn_skip0_4(self.transposeconv_skip0_4(skip0_3))

        #ag4
        ag4_cnn1 = self.ag_cnn4(out3)
        ag4_cnn2 = self.ag_cnn4(skip0_4)
        ag4_alpha1 = ag4_cnn1 + ag4_cnn2
        ag4_cnn3 = self.ag_cnn4(self.relu(ag4_alpha1))
        ag4_alpha2 = self.sigmoid(ag4_cnn3)
        ag4_out = torch.mul(out3, ag4_alpha2)
        ######

        resout_up3 = self.transposeconv_resout_up3(res_encoder_output[-4])
        out4 = torch.cat([ag4_out, resout_up3], dim=1)
        out4 = self.stage1_de(out4)
        out4 = self.transposeconv_stage0(out4)
        cbr_skip = self.cbr2(self.cbr1(res_encoder_output[-5]))

        #ag5
        ag5_cnn1 = self.ag_cnn5(out4)
        ag5_cnn2 = self.ag_cnn5(cbr_skip)
        ag5_alpha1 = ag5_cnn1 + ag5_cnn2
        ag5_cnn3 = self.ag_cnn5(self.relu(ag5_alpha1))
        ag5_alpha2 = self.sigmoid(ag5_cnn3)
        ag5_out = torch.mul(out4, ag5_alpha2)
        ######

        out5 = self.stage0_de(ag5_out)
        out_final = self.sigmoid(self.cls_conv(out5))


        return out_final

    class CAL_Net(nn.Module):
        def __init__(self, backbone_mask, num_classes=2):
            super().__init__()
            # self.oi_encode = backbone_oi
            self.mask_encode = backbone_mask
            # self.zoom_encode = backbone_zoom
            # self.feature_oi = None
            in_dim = 1536
            self.conv1 = nn.Conv3d(in_dim, 512, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(512)
            self.relu = nn.ReLU()

            # self.query_conv = nn.Conv3d(in_dim, in_dim // 8, kernel_size=1)
            # self.key_conv = nn.Conv3d(in_dim, in_dim // 8, kernel_size=1)

            # if backbone_oi.model_type == "ResNet":
            #     value_in_dim = 512
            # elif backbone_oi.model_type == "DenseNet":
            #     value_in_dim = 1024
            # elif backbone_oi.model_type == "EfficientNetBN":
            #     value_in_dim = 1280
            # else:
            #     raise NotImplementedError(backbone_oi.model_type)
            value_in_dim = 512
            self.value_conv = nn.Conv3d(value_in_dim, value_in_dim, kernel_size=1)

            self.softmax = nn.Softmax(dim=-1)
            # self.gamma = nn.Parameter(torch.zeros(1))

            self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
            self.fc = nn.Linear(in_features=value_in_dim, out_features=num_classes, bias=True)

            self.transconv = nn.ConvTranspose3d(768, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)

        def forward(self, x, y, z):
            # encode
            feature_re = x
            feature_tf = self.transconv(y)
            # feature_zoom = self.zoom_encode(y)[-2]
            feature_mask = self.mask_encode(z)[-1]
            features = torch.cat([feature_re, feature_tf, feature_mask], dim=1)
            features = self.conv1(features)
            features = self.bn1(features)
            features = self.relu(features)
            # decode
            # batch_size, channels, depth, height, width  = feature_oi.size()
            # proj_query = self.query_conv(feature_mask).view(batch_size, -1, width * height * depth).permute(0, 2, 1)  # B X CX(N)
            # proj_key = self.key_conv(feature_zoom).view(batch_size, -1, width * height * depth)  # B X C x (*W*H)
            # energy = torch.bmm(proj_query, proj_key)  # transpose check
            # attention = self.softmax(energy)  # BX (N) X (N)
            # proj_value = self.value_conv(feature_oi).view(batch_size, -1, width * height * depth)  # B X C X N
            # att_x = torch.bmm(proj_value, attention.permute(0, 2, 1))
            # att_x = att_x.view(batch_size, channels, depth, height, width)
            # att_x = self.gamma * att_x + feature_oi
            # print('gama\n', self.gamma)
            # att_x = att_x + feature_oi
            # self.finalconv = att_x.clone()
            # classification head
            res = self.avgpool(features)
            res = res.view(res.size(0), -1)
            res = self.fc(res)

            return res