import torch
import torch.nn as nn
from layers.conv_layer import ConvLayer
from layers.activation import get_activation
from layers.normalization import get_norm
from attention.channel_attention import ChannelAttention
from attention.spatial_attention import SpatialAttention

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, stride=stride)
        self.norm1 = get_norm('batch', out_channels)
        self.act1 = get_activation('relu')

        self.conv2 = ConvLayer(out_channels, out_channels)
        self.norm2 = get_norm('batch', out_channels)

        self.ca = ChannelAttention(out_channels, reduction)
        self.sa = SpatialAttention()

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                ConvLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                get_norm('batch', out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # CBAM
        out = self.ca(out)
        out = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
