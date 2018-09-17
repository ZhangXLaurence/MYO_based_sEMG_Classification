import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

import math


class RingLoss(nn.Module):
    """
    Paper:[Ring Loss]
    Ring loss: Convex Feature Normalization for Face Recognition
    """
    def __init__(self, loss_weight=1.0):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight

    def forward(self, feat):
        features = feat.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] == -1: # Initialize as mean norms
            self.radius.data.fill_(features.mean().item())
        diff = features.sub(self.radius.expand_as(features))
        diff_sq = torch.pow(torch.abs(diff), 2).mean() # The only operation that needs gradients.
        return self.loss_weight * diff_sq

class ClasswiseRingLoss(nn.Module):
    """
    Paper:[NaN]
    """
    def __init__(self, num_classes, loss_weight=1.0):
        super(ClasswiseRingLoss, self).__init__()
        self.num_classes = num_classes
        self.radius = nn.Parameter(torch.Tensor(self.num_classes))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight

    def forward(self, feat, label):
        features = feat.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] == -1: # Initialize as mean norms
            self.radius.data.fill_(features.mean().item())

        radius_sel = self.radius.index_select(0, label.long())

        diff = features.sub(radius_sel.expand_as(features))
        diff_sq = torch.pow(torch.abs(diff), 2).mean() # The only operation that needs gradients.

        return self.loss_weight * diff_sq

class CenterLoss(nn.Module):
    """
    Paper:[Center Loss]
    A Discriminative Feature Learning Approach for Deep Face Recognition
    """
    def __init__(self, feat_dim, num_classes, loss_weight=0.1):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.loss_weight = loss_weight

    def forward(self, feat, label):
        feat = feat.view(feat.size(0), 1, 1, -1).squeeze()
        centers_sel = self.centers.index_select(0, label.long()) # According to current labels, select centers in need.
        diff = feat.sub(centers_sel)
        logits = torch.sum(torch.pow(diff, 2), dim=1)
        loss = torch.sum(logits, dim=0) / 2.0
        return logits, self.loss_weight * loss
