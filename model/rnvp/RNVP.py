import torch
from torch import nn
from model.rnvp.CouplingLayer import CouplingLayer
import numpy as np
import utils


# todo generalize for n coupling layer in the network
# todo generalize for more than dimensions of points


class RNVP(nn.Module):
    """
    RNVP network model for 2 layers, using CouplingLayer2
    """

    def __init__(self, input_size, d):
        super().__init__()

        layer_0 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=True)
        layer_1 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=False)
        layer_2 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=True)

        # self.squeeze = nn.Conv2d(1, 4, kernel_size=3, padding=1)

        layer_3 = CouplingLayer(4, 128, mask_type='channel_wise', reverse=False)
        layer_4 = CouplingLayer(4, 128, mask_type='channel_wise', reverse=True)
        layer_5 = CouplingLayer(4, 128, mask_type='channel_wise', reverse=False)

        # self.de_squeeze1 = nn.Conv2d(4, 2, kernel_size=3, padding=1)

        layer_6 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=True)
        layer_7 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=False)
        layer_8 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=True)
        layer_9 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=False)

        # self.de_squeeze2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        self.input_size = input_size
        self.d = d
        self.layers_check1 = nn.ModuleList([layer_0, layer_1, layer_2])
        self.layers_channel = nn.ModuleList([layer_3, layer_4, layer_5])
        self.layers_check2 = nn.ModuleList([layer_6, layer_7, layer_8, layer_9])

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the RNVP network making the data pass into each coupling layers.
        """
        #x = x[0]
        print('-----enetr RNVP forward-------')
        y = x
        print('datay: ', y.size())
        sum_det_J = torch.zeros(len(x))
        for i in range(len(self.layers_check1)):
            y, det_J = self.layers_check1[i].forward(y)
            sum_det_J += det_J
        print('print after check1: ', y.size())
        y = utils.squeeze(y)
        print('print after squeeze: ', y.size())

        for i in range(len(self.layers_channel)):
            y, det_J = self.layers_channel[i].forward(y)
            sum_det_J += det_J
        print('print after channel: ', y.size())
        y = utils.unsqueeze(y)
        print('print after unsqueeze: ', y.size())
        for i in range(len(self.layers_check2)):
            y, det_J = self.layers_check2[i].forward(y)
            sum_det_J += det_J
        print('print after check2: ', y.size())
        return y, sum_det_J

    def inverse(self, y: torch.Tensor):
        """
        Inverse pass of the RNVP network making the data go back through each coupling layers.
        """
        x = y
        sum_det_J = torch.zeros(len(y))

        for i in range(1, len(self.layers_check2) + 1):
            x, det_J = self.layers_check2[len(self.layers_check2) - i].inverse(x)
            sum_det_J += det_J

        x = utils.squeeze(x)

        for i in range(1, len(self.layers_channel) + 1):
            x, det_J = self.layers_channel[len(self.layers_channel) - i].inverse(x)
            sum_det_J += det_J

        x = utils.unsqueeze(x)

        for i in range(1, len(self.layers_check1) + 1):
            x, det_J = self.layers_check1[len(self.layers_check1) - i].inverse(x)
            sum_det_J += det_J

        return x, sum_det_J
