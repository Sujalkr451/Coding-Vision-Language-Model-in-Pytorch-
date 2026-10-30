import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv = nn.Sequential(
            # Standard Conv: 224 -> 112
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Depthwise Conv: 112 -> 112
            nn.Conv2d(32, 32, 3, stride=1, padding=1,
                      groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Pointwise Conv: 32 -> 64 channels
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Depthwise Conv: 112 -> 56
            nn.Conv2d(64, 64, 3, stride=2, padding=1,
                      groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Pointwise Conv: 64 -> 128 channels
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Depthwise Conv: 56 -> 56
            nn.Conv2d(128, 128, 3, stride=1, padding=1,
                      groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)

            


        )