import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self, n, m):
        """

        :param n: length of input
        :param m: length of output
        """
        super(ResNet, self).__init__()


    def forward(self, x):
        return x