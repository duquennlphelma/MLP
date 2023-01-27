import torch
from torch import nn
from model.rnvp.CouplingLayer import CouplingLayer
import utils


class RNVP(nn.Module):
    """
    RNVP network model adapted to image transformation using CouplingLayer.
    """
    def __init__(self, input_size=1, d=64):
        """RNVP constructor for one scale : 3 checkerboard, 4 channel-wise and 4 checkerboard
        :param input_size: number of channels for the input image dataset (1 for MNIST)
        :param d: size of the modified part of the tensor into the CouplingLayer
        """
        super().__init__()

        layer_0 = CouplingLayer(input_size, d, mask_type='checkerboard', reverse=True)
        layer_1 = CouplingLayer(input_size, d, mask_type='checkerboard', reverse=False)
        layer_2 = CouplingLayer(input_size, d, mask_type='checkerboard', reverse=True)

        layer_3 = CouplingLayer(4*input_size, 2*d, mask_type='channel_wise', reverse=False)
        layer_4 = CouplingLayer(4*input_size, 2*d, mask_type='channel_wise', reverse=True)
        layer_5 = CouplingLayer(4*input_size, 2*d, mask_type='channel_wise', reverse=False)

        layer_6 = CouplingLayer(input_size, d, mask_type='checkerboard', reverse=True)
        layer_7 = CouplingLayer(input_size, d, mask_type='checkerboard', reverse=False)
        layer_8 = CouplingLayer(input_size, d, mask_type='checkerboard', reverse=True)
        layer_9 = CouplingLayer(input_size, d, mask_type='checkerboard', reverse=False)

        self.input_size = input_size
        self.d = d
        self.layers_check1 = nn.ModuleList([layer_0, layer_1, layer_2])
        self.layers_channel = nn.ModuleList([layer_3, layer_4, layer_5])
        self.layers_check2 = nn.ModuleList([layer_6, layer_7, layer_8, layer_9])

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the RNVP network making the data pass into each coupling layers.
        """
        y = x
        sum_det_J = torch.zeros(len(y))

        for i in range(len(self.layers_check1)):
            y, det_J = self.layers_check1[i].forward(y)
            sum_det_J += det_J

        y = utils.squeeze(y)

        for i in range(len(self.layers_channel)):
            y, det_J = self.layers_channel[i].forward(y)
            sum_det_J += det_J

        y = utils.unsqueeze(y)

        for i in range(len(self.layers_check2)):
            y, det_J = self.layers_check2[i].forward(y)
            sum_det_J += det_J

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
