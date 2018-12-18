import torch
import torch.nn as nn

class immitation_model(nn.Module):
    def __init__(self):
        super(immitation_model, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride = 2, bias = False),
            nn.BatchNorm2d(32),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, stride = 1, bias = False),
            nn.BatchNorm2d(32),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride = 2, bias = False),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride = 1, bias = False),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride = 2, bias = False),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride = 1, bias = False),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride = 1, bias = False),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride = 1, bias = False),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(0.35),
            nn.ReLU()
        )

        self.speed_net = nn.Sequential(
            nn.Linear(8192, 1, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )


        self.acceleration_net = nn.Sequential(
            nn.Linear(8192, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )


        self.steer_net = nn.Sequential(
            nn.Linear(8192, 1, bias = False),
            # nn.ReLU(),
            # nn.Linear(100, 50, bias = False),
            # nn.ReLU(),
            # nn.Linear(50, 10, bias = False),
            # nn.ReLU(),
            # nn.Linear(10, 1, bias = False)
        )

        self.break_net = nn.Sequential(
            nn.Linear(8192, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 8192)
        out = self.steer_net(x)
        return out
