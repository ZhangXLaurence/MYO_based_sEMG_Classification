import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=10, feature_dim=64, dense_dim=32):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3)
        self.prelu0 = nn.PReLU()
        self.fc1 = nn.Linear(4*8, dense_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(dense_dim, dense_dim)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(dense_dim, dense_dim)
        self.prelu3 = nn.PReLU()
        self.ip1 = nn.Linear(dense_dim, feature_dim)
        # self.ip_relu = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu0(x.view(-1, 4*8))
        x1 = self.prelu1(self.fc1(x))
        x2 = self.prelu2(self.fc2(x1))
        x3 = self.prelu3(self.fc3(x2 + x1))
        ip1 = self.ip1(x3+x2+x1)    # feature
        # ip1 = self.ip_relu(ip1)
        return ip1



class SmallNet(nn.Module):
    def __init__(self, feature_dim):
        super(SmallNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, feature_dim)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        ip1 = self.preluip1(self.ip1(x))
        # ip2 = self.ip2(ip1)
        return ip1
