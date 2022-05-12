### model Creation of ResNet20
import torch
from torch import nn


class SkipConnection(nn.Module):

    def __init__(self, f, c=nn.Identity()):
        """
        Description
        """
        super().__init__()
        self.f = f
        self.c = c
        self.ff = nn.quantized.FloatFunctional()
        
    def forward(self, X):
        """
        Description
        """
        return self.ff.add_relu(self.c(X), self.f(X))

def ResNet20():
    return nn.Sequential(
        ### Initial Layer
        nn.Conv2d(3, 16, 3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        ### Skip Connections
        *[SkipConnection(
                nn.Sequential(
                    nn.Conv2d(16, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                )
        ) for _ in range(3)],
        SkipConnection(
            nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(32),
            ),
        ),

        *[SkipConnection(
                nn.Sequential(
                    nn.Conv2d(32, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
        ) for _ in range(3)],
        SkipConnection(
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
            ),
        ),
        
        *[SkipConnection(
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                )
        ) for _ in range(3)],
    
        ### Flattening
        nn.AvgPool2d(8),
        nn.Flatten(start_dim=1, end_dim=-1),
        
        ### Head Layer
        nn.Linear(64, 10)
    )