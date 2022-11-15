import torch
from torch import nn

# todo: generalization for more dimensions
# question: is it better to write an option or permute the dimensions as an option
#    of coupling layer class?

# global variables
D = 2  # nr of dimensions (for an image its 32^2 or 64^2)
d = 1  # nr of dimensions unchanged (identity) through a coupling layer


class CouplingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional ResNets or DenseNets with skip connections
        # and rectifier non-linearities
        self.s = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, D - d)
        )
        self.t = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, D - d)
        )

    def forward(self, x):
        coupling_out = torch.cat([x[0:d], torch.mul(x[d:D], torch.exp(self.s(x[0:d]))) + self.t(x[0:d])])
        return coupling_out


class RNVP(nn.Module):
    # abbreviation of real valued non-volume preserving transformations
    def __init__(self, nr_of_coupling_layers=2, split_and_permutations_of_dimensions=1):
        super().__init__()

        layer_0 = CouplingLayer()
        self.layer_0 = layer_0

        layer_1 = CouplingLayer()
        self.layer_1 = layer_1

        self.permutation = torch.Tensor([[0, 1], [1, 0]])

    def forward(self, x):
        y = self.layer_0.forward(x)
        y_shuffled = self.permutation @ y
        z = self.layer_1.forward(y_shuffled)
        return z
