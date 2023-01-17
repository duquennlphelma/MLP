import torch
from torch import nn
from model.rnvp.CouplingLayer import CouplingLayer


# todo generalize for n coupling layer in the network
# todo generalize for more than dimensions of points


class RNVP(nn.Module):
    """
    RNVP network model for 2 layers, using CouplingLayer2
    """

    def __init__(self, input_size, d):
        super().__init__()

        layer_0 = CouplingLayer(input_size, d, reverse=True)
        layer_1 = CouplingLayer(input_size, d, reverse=False)
        layer_2 = CouplingLayer(input_size, d, reverse=True)
        layer_3 = CouplingLayer(input_size, d, reverse=False)
        layer_4 = CouplingLayer(input_size, d, reverse=True)
        layer_5 = CouplingLayer(input_size, d, reverse=False)

        self.input_size = input_size
        self.d = d

        self.layers = nn.ModuleList([layer_0, layer_1, layer_2, layer_3, layer_4, layer_5])

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the RNVP network making the data pass into each coupling layers.
        """
        y = x
        #sum_det_J is size batch_size,1
        sum_det_J = torch.zeros(len(x))
        for i in range(len(self.layers)):
            y, det_J = self.layers[i].forward(y)

            # summing s_x from all the coupling layers : that's LDJ : log determinant jacobian
            sum_det_J += det_J

        return y, sum_det_J

    def inverse(self, y: torch.Tensor):
        """
        Inverse pass of the RNVP network making the data go back through each coupling layers.
        """

        x = y
        sum_det_J = torch.zeros(len(y))
        for i in range(1, len(self.layers)+1):
            x, det_J = self.layers[len(self.layers)-i].inverse(x)
            sum_det_J += det_J

        return x, sum_det_J
