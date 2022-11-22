import torch
from torch import nn
from model.rnvp.CouplingLayer2 import CouplingLayer

# next: generalization for more dimensions


class RNVP(nn.Module):
    # abbreviation of real valued non-volume preserving transformations
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
        Data goes through each of coupling layers.
        Between layers it is permutated,
        so that non dimension goes through identities only,
        therefore each dimension is transformed.
        """
        y_mid = self.layer_0.forward(x)
        y_out = self.layer_1.forward(y_mid)
        return y_out
