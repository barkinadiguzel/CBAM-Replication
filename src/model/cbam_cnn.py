import torch
import torch.nn as nn
from backbone.resnet_blocks import BasicBlock

class CBAM_CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CBAM_CNN, self).__init__()
        self.layer1 = BasicBlock(in_channels, 64)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
