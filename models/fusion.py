import torch
import torch.nn as nn

class MAFModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, ir):
        fused = torch.cat([rgb, ir], dim=1)
        attention = self.attn(fused)
        return fused * attention
