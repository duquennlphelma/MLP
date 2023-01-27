import torch
from torch import nn
import utils


class CouplingLayer(nn.Module):
    def __init__(self, input_channels, d_channels, mask_type, reverse=False):
        """
        Initialisation of the coupling layer
        Case of datasets of images
        :param input_channels: number of channels in the input
        :param d_channels: size of the modified part of the tensor into the CouplingLayer -> number of channels in s & t
        :param mask_type: mask type (checkerboard or channel-wise)
        :param reverse: whether to reverse the mask
        """
        super().__init__()

        self.reverse = reverse
        self.d = d_channels
        self.input_size = input_channels
        self.mask_type = mask_type

        conv1 = nn.Conv2d(input_channels, d_channels, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(d_channels, input_channels, kernel_size=3, padding=1)
        norm_in = nn.BatchNorm2d(input_channels)
        norm_out = nn.BatchNorm2d(input_channels)

        list_t = [norm_in, conv1, nn.ReLU(),  conv2, norm_out]
        self.t = nn.Sequential(*list_t)
        list_s = [norm_in, conv1, nn.ReLU(), conv2, norm_out, nn.Tanh()]
        self.s = nn.Sequential(*list_s)

    def forward(self, x):
        """
        Definition of the forward pass into a coupling layer for an input x
        """
        size = torch.Tensor.size(x)  # returns (batch_size, n_channels, h, w)

        if self.mask_type == 'checkerboard':
            b = utils.checkerboard_mask(size[-2], size[-1], reverse_mask=self.reverse)
        if self.mask_type == 'channel_wise':
            b = utils.channel_mask(self.input_size, reverse_mask=self.reverse)

        b_x = torch.mul(x, b)
        s_x = self.s(b_x) * (1-b)
        t_x = self.t(b_x) * (1-b)

        z = (1 - b) * (x - t_x) * torch.exp(-s_x) + b_x

        det_J = s_x.view(s_x.size(0), -1).sum(-1)

        return z, det_J

    def inverse(self, y):
        """
        Definition of the inverse pass into a coupling layer for an output y
        """
        size = torch.Tensor.size(y)  # returns (batch_size, n_channels, h, w)

        if self.mask_type == 'checkerboard':
            b = utils.checkerboard_mask(size[-2], size[-1], reverse_mask=self.reverse)
        if self.mask_type == 'channel_wise':
            b = utils.channel_mask(self.input_size, reverse_mask=self.reverse)

        b_x = torch.mul(y, b)
        s_x = self.s(b_x)
        t_x = self.t(b_x)

        z = b_x + (1 - b) * (y * torch.exp(s_x) + t_x)

        det_J = (-s_x).view(s_x.size(0), -1).sum(-1)

        return z, det_J


