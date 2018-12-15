import torch
import torch.nn as nn

class base_model(nn.Module):
    def __init__(self):
        super(base_model, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride = 2, bias = False),
            nn.ReLU(),

            nn.Conv2d(24, 36, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(36),

            nn.Conv2d(36, 48, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 64, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.Dropout(p=0.35)
        )

        self.linear_net = nn.Sequential(
            nn.Linear(64 * 22 * 15, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 2, bias = False)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 64 * 15 * 22)
        x = self.linear_net(x)
        return x

