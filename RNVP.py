import torch
from torch import nn
from CouplingLayer import CouplingLayer, d
from data_simple import D
# next: generalization for more dimensions


class RNVP(nn.Module):
    # abbreviation of real valued non-volume preserving transformations
    def __init__(self, nr_of_coupling_layers=2, split_and_permutations_of_dimensions=1):
        super().__init__()

        layer_0 = CouplingLayer(D,d)
        self.layer_0 = layer_0

        layer_1 = CouplingLayer(D,d)
        self.layer_1 = layer_1

        self.permutation = torch.Tensor([[0, 1], [1, 0]])

    def forward(self, x):
        y = self.layer_0.forward(x)
        y_shuffled = self.permutation @ y
        z = self.layer_1.forward(y_shuffled)
        return z
