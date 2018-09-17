import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

import math

class NormalizedInnerProductWithScale(nn.Module):
    """
    Paper:[COCOv2]
    Rethinking Feature Discrimination and Polymerization for Large scale recognition
    """
    def __init__(self, feature_dim, class_num, scale=20):
        super(NormalizedInnerProductWithScale, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        return logits

class ArcFaceInnerProduct(nn.Module):
    """
    Paper:[ArcFace]
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    """
    def __init__(self, feature_dim, class_num, scale=30.0, margin=0.5, easy_margin=False):
        super(ArcFaceInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
        # Setting behavior according to the value of easy_margin
        if easy_margin == -1: # If easy_margin = -1, just as the statement in the paper.
            self.Threshholder = -1
            self.out_indicator = 0
        elif easy_margin is True: # If easy_margin is True, for angle bigger than pi/2, do not add margin.
            self.Threshholder = 0
            self.out_indicator = 0
        else: # If easy_margin is Flase, do some adjustment for adding margin.
            self.Threshholder = - math.cos(self.margin)
            self.out_indicator = 1

    def forward(self, feat, label, return_ip=False):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        innerproduct_logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables_ext = torch.zeros_like(cos)
        margin_tables_ext = Variable(margin_tables_ext).cuda()
        thetas = []
        for i in range(margin_tables.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
            avg_theta = sum(thetas) / len(thetas)
            max_theta = max(thetas)
            min_theta = min(thetas) 
            print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))
            if cos[i, label_i].item() > self.Threshholder:
                margin_tables[i, label_i] += self.margin
            else:
                margin_tables_ext[i, label_i] -= self.margin * math.sin(self.margin)
        # Calculate marginal logits
        margin_tables_ext *= self.out_indicator
        marginal_logits = self.scale * (torch.cos(torch.acos(cos) + margin_tables) + margin_tables_ext)
        if return_ip:
            return cos, marginal_logits, innerproduct_logits
        else:
            return cos, marginal_logits
