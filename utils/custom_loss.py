import torch
import torch.nn.functional as F
from torch import nn


class CustomCrossEntropyLoss(nn.Module):
    """
    Custom Cross Entropy Loss with Non-uniform Label Smoothing
    """

    def __init__(self, label_smoothing=0.0, reduction='mean', class_num=10):
        super(CustomCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, input, target):
        smooth_labels = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1.0)

        if self.label_smoothing > 0:
            smooth_labels = smooth_labels * (1 - self.label_smoothing)
            rand_labels = torch.rand_like(smooth_labels) * self.label_smoothing
            mask = smooth_labels.bool()
            rand_labels = rand_labels.masked_fill_(mask, 0)
            rand_labels_sum = rand_labels.sum(dim=1, keepdim=True)
            rand_labels = rand_labels / rand_labels_sum * self.label_smoothing
            smooth_labels += rand_labels

        log_probs = F.log_softmax(input, dim=-1)
        loss = (-smooth_labels * log_probs).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
