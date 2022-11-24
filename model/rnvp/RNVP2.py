import torch
from torch import nn
from model.rnvp.CouplingLayer2 import CouplingLayer

#todo generalize for n coupling layer in the network
#todo use learnable function for s and t
#todo generalize for more than dimensions of points


class RNVP(nn.Module):
    """
    RNVP network model for 2 layers, using CouplingLayer2
    """
    def __init__(self, input_size, d):
        super().__init__()

        layer_0 = CouplingLayer(input_size, d, up=True)
        self.layer_0 = layer_0

        layer_1 = CouplingLayer(input_size, d, up=False)
        self.layer_1 = layer_1

        self.input_size = input_size
        self.d = d

    def forward(self, x: torch.Tensor):
        """
        Forword pass of the RNVP network making the data pass into each coupling layers.
        """
        y_mid = self.layer_0.forward(x)
        y_out = self.layer_1.forward(y_mid)
        return y_out
