import torch
from torch import nn
from model.rnvp.CouplingLayer2 import CouplingLayer


class RNVP(nn.Module):
    """
    RNVP network model adapted to 2D Dataset transformation using CouplingLayer2.
    """

    def __init__(self, input_size=2, d=1):
        """RNVP constructor for one scale : 3 checkerboard, 4 channel-wise and 4 checkerboard
        :param input_size: size of an input sample (2 for Moon & Fun)
        :param d: size of the modified part of the tensor into the CouplingLayer
        """
        super().__init__()

        layer_0 = CouplingLayer(input_size, d, up=True)
        layer_1 = CouplingLayer(input_size, d, up=False)
        layer_2 = CouplingLayer(input_size, d, up=True)
        layer_3 = CouplingLayer(input_size, d, up=False)
        layer_4 = CouplingLayer(input_size, d, up=True)
        layer_5 = CouplingLayer(input_size, d, up=False)

        self.input_size = input_size
        self.d = d

        self.layers = nn.ModuleList([layer_0, layer_1, layer_2, layer_3, layer_4, layer_5])

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the RNVP network making the data pass into each coupling layers.
        """
        y = x
        sum_det_J = torch.zeros(len(x))  # sum_det_J is size (batch_size, 1)

        for i in range(len(self.layers)):
            y, det_J = self.layers[i].forward(y)
            sum_det_J += det_J  # summing s_x from all the coupling layers : that's LDJ : log determinant Jacobian

        return y, sum_det_J

    def inverse(self, y: torch.Tensor):
        """
        Inverse pass of the RNVP network making the data go back through each coupling layers.
        """
        x = y
        sum_det_J = torch.zeros(len(y))

        for i in range(1, len(self.layers) + 1):
            x, det_J = self.layers[len(self.layers) - i].inverse(x)
            sum_det_J += det_J

        return x, sum_det_J
