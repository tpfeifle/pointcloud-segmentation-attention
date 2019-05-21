"""
This module will contain the attention layers we create
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.nn.MultiheadAttention

class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()

    def forward(self, x):
        """

        :param x: (B x N x D)
        :return:
        """
