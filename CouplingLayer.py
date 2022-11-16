import torch
from torch import nn
from data_simple import D

d = 1  # nr of dimensions unchanged (identity) through a coupling layer


# next: ResNets for s and t
# question: is it better to write an option to permute the dimensions in the coupling layer class?

class CouplingLayer(nn.Module):
    def __init__(self, D, d):
        super().__init__()
        # convolutional ResNets or DenseNets with skip connections
        # and rectifier non-linearities
        self.s_recent = None
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
        print(self, "Trainable parameters are: ")
        for param in self.parameters():
            print("parameter shape: ", param.shape)

    def forward(self, x):
        self.s_recent = self.s(x[0:d])    # will be used in loss function
        coupling_out = torch.cat([self.s_recent, torch.mul(x[d:D], torch.exp(self.s_recent) + self.t(x[0:d]))])
        return coupling_out
