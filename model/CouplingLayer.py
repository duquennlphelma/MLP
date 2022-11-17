import torch
from torch import nn
from model.ResNet import ResNet


# next: ResNets for s and t
# question: is it better to write an option to permute the dimensions in the coupling layer class?
# define D using shape of data variable
# define d to be a roof of half of D?

class CouplingLayer(nn.Module):
    def __init__(self, D, d):   # length of tensor, length of unchanged part
        super().__init__()
        # todo: convolutional ResNets or DenseNets with skip connections
        #   and rectifier non-linearities for s and t
        self.s_recent = None
        self.s = ResNet(d, D-d)    # dimension of input, dimension of output
        self.t = ResNet(d, D-d)
        self.d = d
        self.D = D
        print(self, "Trainable parameters are: ")
        for param in self.parameters():
            print("parameter shape: ", param.shape)

    def forward(self, x):
        self.s_recent = self.s(x[0:self.d])    # will be used in loss function
        coupling_out = torch.cat([self.s_recent, torch.mul(x[self.d:self.D], torch.exp(self.s_recent) + self.t(x[0:self.d]))])
        return coupling_out
