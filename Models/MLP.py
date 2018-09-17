import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=10, feature_dim=64, dense_dim=32):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, dense_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(dense_dim, dense_dim)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(dense_dim, dense_dim)
        self.prelu3 = nn.PReLU()
        self.fc4 = nn.Linear(dense_dim, dense_dim)
        self.prelu4 = nn.PReLU()
        self.ip1 = nn.Linear(dense_dim, feature_dim)
        # self.ip2 = nn.Linear(feature_dim, class_num)

    def forward(self, x):
        x1 = self.prelu1(self.fc1(x))
        x2 = self.prelu2(self.fc2(x1))
        x3 = self.prelu3(self.fc3(x2 + x1))
        x4 = self.prelu4(self.fc4(x3 + x2 + x1))
        ip1 = self.ip1(x4)    # feature
        # ip2 = self.ip2(ip1)   # logit
        return ip1