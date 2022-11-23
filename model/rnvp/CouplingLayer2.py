import torch
from torch import nn
from model.resnet.ResNet import ResNet
import torch.nn.functional
import numpy as np


class CouplingLayer(nn.Module):
    def __init__(self, input_size, d, up=True):
        """
        Initialisation of the coupling layer
        Specific case of dataset of non images but points with [x, y] coordinates.
        :param input_size: size of the input
        :param output_size: size of the output
        """
        super().__init__()
        # todo: convolutional ResNets or DenseNets with skip connections
        #   and rectifier non-linearities for s and t

        self.s = nn.Linear(input_size, input_size - d)
        self.t = nn.Linear(input_size, input_size - d)
        self.d = d
        self.input_size = input_size
        if up:
            self.mask = torch.FloatTensor(np.concatenate((np.ones(d), np.zeros(input_size - d)), axis=None))
        else:
            self.mask = torch.FloatTensor(np.concatenate((np.zeros(d), np.ones(input_size - d)), axis=None))

    def forward(self, x):
        """
        Definition of the forward pass into a coupling layer for an input x
        :param x: input of the layer
        :return: output of the layer
        """
        b = self.mask

        b_x = torch.mul(x, b)
        s_x = self.s(b_x)
        t_x = self.t(b_x)

        y = b_x + torch.mul((1-b), (torch.mul(x, torch.exp(s_x)) + t_x))

        return y
