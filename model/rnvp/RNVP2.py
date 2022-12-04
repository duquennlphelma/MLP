import torch
from torch import nn
from model.rnvp.CouplingLayer2 import CouplingLayer

#todo generalize for n coupling layer in the network
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

        self.layers= nn.ModuleList([self.layer_0,self.layer_1])

    def forward(self, x: torch.Tensor):
        """
        Forword pass of the RNVP network making the data pass into each coupling layers.
        """
        y=x
        sum_det_J=0
        for i in range(len(self.layers)):
            y, det_J=self.layers[i].forward(y)
            sum_det_J=sum_det_J+ det_J

        print('sum detJ \n', sum_det_J)

        return y, sum_det_J

    def inverse(self, y:torch.Tensor):
        sum_det_J=0
        x=y
        for i in range(len(self.layers),-1,-1):
            x, det_J=self.layers[i].forward(x)
            sum_det_J=sum_det_J+det_J

        return x, sum_det_J

