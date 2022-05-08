### model Creation of ResNet20
import torch
from torch import nn

class SkipConnection(nn.Module):

    def __init__(self, f_m, f_s=None):
        """
        Description
        """
        super().__init__()
        self.f_m = f_m
        self.f_s = f_s
        self.relu = nn.ReLU()
        
    def forward(self, X):
        """
        Description
        """
        if self.f_s is not None:
            return self.relu(self.f_s(X) + self.f_m(X))
        else:
            return self.relu(X + self.f_m(X))

class BlockConnection(nn.Module):

    def __init__(self, channels, n_blocks):
        """
        Description
        """
        super().__init__()
        self.connections = nn.ModuleList([
            SkipConnection(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                )   
            ) for _ in range(n_blocks)])

    def forward(self, X):
        """
        Description
        """
        out = X
        for i, module in enumerate(self.connections):
            out = module(out)

        return out

class DownsamplingConnection(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Description
        """
        super().__init__()
        self.module =  SkipConnection(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
        )

    def forward(self, X):
        """
        Description
        """
        return self.module(X)

def model():
    return nn.Sequential(
        ### Initial Layer
        nn.Conv2d(3, 16, 3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        ### Skip Connections
        BlockConnection(16, 3),
        DownsamplingConnection(16, 32),

        BlockConnection(32, 2),
        DownsamplingConnection(32, 64),

        BlockConnection(64, 2),
    
        ### Flattening
        nn.AvgPool2d(8),
        nn.Flatten(start_dim=1, end_dim=-1),
        
        ### Head Layer
        nn.Linear(64, 10)
    )