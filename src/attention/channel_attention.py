import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()
        avg_pool = torch.mean(x, dim=(2,3), keepdim=False)  
        max_pool,_ = torch.max(x, dim=(2,3))                

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out
        out = self.sigmoid(out).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

        return x * out
