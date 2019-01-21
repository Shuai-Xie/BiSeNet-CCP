import torch
from torch import nn
from model.build_contextpath import build_contextpath


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.bn(self.conv1(input)))  # CBR


class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    # 3 layers
    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


# ARM
class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # in = out
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels

    def forward(self, input):
        # global average pooling
        x = torch.mean(input, dim=3, keepdim=True)  # first cols get mean, then rows get mean
        x = torch.mean(x, dim=2, keepdim=True)  # fm -> one value, fms -> vector
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)  # when test, no need BN
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


# FFM
class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        self.in_channels = 3032  # 256 + (1024 + 2048) = 3328
        # ConvBlock 3x3 kernel
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)  # 3328->151
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        # global average pool
        x = torch.mean(feature, 3, keepdim=True)
        x = torch.mean(x, 2, keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module
        self.attention_refinement_module1 = AttentionRefinementModule(728, 728)  # in = out
        self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)

        # build feature fusion module
        self.feature_fusion_module = FeatureFusionModule(num_classes)

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)  # use RGB channels on SP

        # output of context path
        cx1, cx2, tail = self.context_path(input)  # tail 1/32 fm global average
        # print(cx1.shape)  # torch.Size([1, 728, 30, 40])
        # print(cx2.shape)  # torch.Size([1, 2048, 15, 20])

        cx1 = self.attention_refinement_module1(cx1)  # 1/16
        cx2 = self.attention_refinement_module2(cx2)  # 1/32
        cx2 = torch.mul(cx2, tail)

        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)  # context path output (2048+1024) 1/8, 1/8

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')  # 1,1
        result = self.conv(result)
        return result
