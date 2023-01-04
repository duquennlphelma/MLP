import torch
from torch import nn
from model.resnet.ResNet import ResNet
import torch.nn.functional
import numpy as np

#todo use learnable function for s and t
#todo generalize for more than dimensions of points


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
        list_f = [nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size - d)]
        #list_f = [nn.Sequential(nn.Linear(input_size, input_size), nn.LeakyReLU(), nn.Linear(input_size, input_size), nn.LeakyReLU(), nn.Linear(input_size, input_size-d),
        #             nn.Tanh())]
        list_v = [nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size - d)]
        #list_v = [nn.Sequential(nn.Linear(input_size, input_size), nn.LeakyReLU(), nn.Linear(input_size, input_size), nn.LeakyReLU(), nn.Linear(input_size, input_size-d))]
        self.s = nn.Sequential(*list_v)
        self.t = nn.Sequential(*list_f)
        self.d = d
        self.input_size = input_size
        if up:
            self.mask = torch.FloatTensor(np.concatenate((np.ones(d), np.zeros(input_size - d)), axis=None))
        else:
            self.mask = torch.FloatTensor(np.concatenate((np.zeros(d), np.ones(input_size - d)), axis=None))

    def forward(self, x):
        """
        Definition of the forward pass into a coupling layer for an input x
        :param x: input of the layer
        :return: output of the layer
        """
        x = torch.Tensor(x)
        b = self.mask

        b_x = torch.mul(x, b)
        s_x = self.s(b_x)
        t_x = self.t(b_x)


        y = b_x + torch.mul((1-b), (torch.mul(x, torch.exp(s_x)) + t_x))

        #s_x is a function from d in d and we are summing over d
        det_J= torch.sum(s_x, -1)

        return y, det_J

    def inverse(self,y):
        y = torch.Tensor(y)
        b = self.mask

        b_x = torch.mul(y, b)
        s_x = self.s(b_x)
        t_x = self.t(b_x)

        x = b_x + torch.mul((torch.mul((1 - b), y) -t_x), torch.exp(-s_x))

        det_J=torch.sum(-s_x)

        return x,det_J


