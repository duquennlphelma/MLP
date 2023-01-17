import torch
from torch import nn
from model.resnet.ResNet import ResNet
import utils
import numpy as np


class CouplingLayer(nn.Module):
    def __init__(self, input_channels, d_channels, reverse=False):
        """
        Initialisation of the coupling layer
        Case of datasets of images
        :param input_channels: size of the input -> number of channels in the input
        :param d_channels: size of the modified part of the tensor into the CouplingLayer -> number of channels in s & t
        :param reverse: whether to reverse the mask
        """
        super().__init__()

        self.reverse = reverse
        self.d = d_channels
        self.input_size = input_channels

        conv1 = nn.Conv2d(input_channels, d_channels, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(d_channels, input_channels, kernel_size=3, padding=1)
        norm_in = nn.BatchNorm2d(input_channels)
        norm_out = nn.BatchNorm2d(input_channels)

        list_t = [norm_in, conv1, nn.ReLU(), conv2, norm_out]
        self.t = nn.Sequential(*list_t)
        list_s = [norm_in, conv1, nn.ReLU(), conv2, norm_out, nn.Tanh()]
        self.s = nn.Sequential(*list_s)

    def forward(self, x):
        """
        Definition of the forward pass into a coupling layer for an input x
        :param x: input of the layer
        :return: output of the layer
        """
        # x = torch.Tensor(x)
        print('x:\n', x)
        x=x[0]
        size = x.size()  # returns (batch_size, n_channels, h, w)
        b = utils.checkerboard_mask(size[-2], size[-1], reverse_mask=self.reverse)

        b_x = torch.mul(x, b)
        s_x = self.s(b_x) * (1-b)
        t_x = self.t(b_x) * (1-b)

        # y = b_x + torch.mul((1-b), (torch.mul(x, torch.exp(s_x)) + t_x))
        z = (1 - b) * (x - t_x) * torch.exp(-s_x) + b_x

        #s_x is a vector of size (batch_size, n_channels, h, w) and we sum on ? to have the determinant for each samples
        #and we took the logarithm of the det that is why we sum the s and log(1)=0
        #det_J = torch.sum(s_x, 1)

        det_J = s_x.view(s_x.size(0), -1).sum(-1)

        return z, det_J

    def inverse(self, y):
        """
        Definition of the inverse pass into a coupling layer for an output y
        :param y: output of the layer
        :return: input of the layer
        """
        # y = torch.Tensor(y)
        b = self.mask

        b_x = torch.mul(y, b)
        s_x = self.s(b_x)
        t_x = self.t(b_x)
        z = b_x + (1 - b) * (y * torch.exp(s_x) + t_x)
        #y = b_x + torch.mul((torch.mul((1 - b), y) - t_x), torch.exp(-s_x))

        det_J = (-s_x).view(s_x.size(0), -1).sum(-1)

        return z, det_J


