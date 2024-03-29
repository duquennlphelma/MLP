import torch
from torch import nn
import torch.nn.functional
import numpy as np


class CouplingLayer(nn.Module):
    def __init__(self, input_size, d, up=True):
        """
        Initialisation of the coupling layer
        Specific case of dataset of non images but points with [x, y] coordinates.
        :param input_size: size of the input
        :param d: size of the modified part of the vector into the CouplingLayer
        :param up: if True the modified part of the vector is the upper part
                   if False the modified part of the vector is the lower part
        """
        super().__init__()

        list_t = [nn.Linear(input_size, 256), nn.LeakyReLU(), nn.Linear(256, 256),
                  nn.LeakyReLU(), nn.Linear(256, input_size)]
        self.t = nn.Sequential(*list_t)

        list_s = [nn.Linear(input_size, 256), nn.LeakyReLU(), nn.Linear(256, 256),
                  nn.LeakyReLU(), nn.Linear(256, input_size), nn.Tanh()]
        self.s = nn.Sequential(*list_s)

        self.d = d
        self.input_size = input_size

        if up:
            self.mask = torch.FloatTensor(np.concatenate((np.ones(d), np.zeros(input_size - d)), axis=None))
        else:
            self.mask = torch.FloatTensor(np.concatenate((np.zeros(d), np.ones(input_size - d)), axis=None))

    def forward(self, x):
        """
        Definition of the forward pass into a coupling layer for an input x
        """
        x = torch.Tensor(x)
        b = self.mask

        b_x = torch.mul(x, b)
        s_x = self.s(b_x) * (1-b)
        t_x = self.t(b_x) * (1-b)

        z = (1 - b) * (x - t_x) * torch.exp(-s_x) + b_x

        # s_x is a vector of size (batch_size, d) and we sum on d to have the determinant for each samples
        # and we took the logarithm of the det that is why we sum the s and log(1)=0
        det_J = torch.sum(s_x, 1)

        return z, det_J

    def inverse(self, y):
        """
        Definition of the inverse pass into a coupling layer for an output y
        """
        y = torch.Tensor(y)
        b = self.mask

        b_x = torch.mul(y, b)
        s_x = self.s(b_x)
        t_x = self.t(b_x)
        z = b_x + (1 - b) * (y * torch.exp(s_x) + t_x)

        det_J = torch.sum(-s_x, 1)

        return z, det_J


