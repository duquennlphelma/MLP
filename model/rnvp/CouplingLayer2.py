import torch
from torch import nn
from model.resnet.ResNet import ResNet


# next: ResNets for s and t
# question: is it better to write an option to permute the dimensions in the coupling layer class?
# define D using shape of data variable
# define d to be a roof of half of D?

class CouplingLayer(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialisation of the coupling layer
        :param input_size: size of the input
        :param output_size: size of the output
        """
        super().__init__()
        # todo: convolutional ResNets or DenseNets with skip connections
        #   and rectifier non-linearities for s and t
        self.s_recent = None
        self.s = ResNet(output_size, input_size-output_size)
        self.t = ResNet(output_size, input_size-output_size)
        self.output_size = output_size
        self.input_size = input_size

        print(self, "Trainable parameters are: ")
        for param in self.parameters():
            print("parameter shape: ", param.shape)

    def forward(self, x):
        """
        Definition of the forward pass into a coupling layer for an input x
        :param x: input of the layer
        :return: output of the layer
        """
        self.s_recent = self.s(x[0:self.output_size])    # will be used in loss function
        coupling_out = torch.cat([self.s_recent, torch.mul(x[self.output_size+1:self.input_size], torch.exp(self.s_recent) + self.t(x[0:self.input_size]))])
        return coupling_out
