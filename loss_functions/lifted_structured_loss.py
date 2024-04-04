import torch
import torch.nn as nn
from  torch import Tensor
import math

class LiftedStructuredLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(LiftedStructuredLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        # Compute pairwise distance between features
        device = features.device
        pairwise_distance = torch.cdist(features, features)

        # Create a mask for positive pairs
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        invert_mask = ~mask
        neg_dists = torch.exp((self.margin - pairwise_distance)) * invert_mask

        mask *= ((torch.ones(mask.shape[0], mask.shape[0]).to(device) - torch.eye(mask.shape[0], mask.shape[0]).to(device) == 1))

        row_exp_sum = torch.sum(neg_dists, dim=1)
        table_exp_sum = torch.max(row_exp_sum.unsqueeze(0), row_exp_sum.unsqueeze(1)) + torch.min(row_exp_sum.unsqueeze(0), row_exp_sum.unsqueeze(1))

        loss = (pairwise_distance + torch.log(table_exp_sum)) * mask
        loss = torch.square(torch.relu(loss))

        loss = torch.sum(loss) / torch.sum(mask)

        return loss
