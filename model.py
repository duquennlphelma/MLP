import torch
from torch import nn

D = 2
d = 1
# TODO: define coupling layers



class CouplingLayer(nn.Module):
    def __init__(self, s, t):
        # Todo: is that way of bringing parameters to the layer right?
        #    both should be global
        super().__init__()
        self.s = s
        self.t = t

    @staticmethod
    def forward(s, t, x, d, D):
        # on secondary dimensions there will be this:
        z = torch.cat(x[0, d], torch.mul(x[d, D], torch.exp(s(x[0, d]))) + t(x[0, d]))
        return z

class RNVP(nn.Module):

    def __init__(self, nr_of_coupling_layers=2, split_and_permutations_of_dimensions=1):
        super().__init__()
        self.s_0 = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, D-d)
            )
        self.t_0 = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, D - d)
        )

        self.s_1 = nn.Sequential(
            nn.Linear(D-d, 128),
            nn.ReLU(),
            nn.Linear(128, d)
        )
        self.t_1 = nn.Sequential(
            nn.Linear(D-d, 128),
            nn.ReLU(),
            nn.Linear(128, d)
        )
        layer_0 = CouplingLayer(self.s_0, self.t_0)
        layer_1 = CouplingLayer(self.s_1, self.t_1)



    def model_forward(thing):

        #for i in range(nr_of_coupling_layers):
            #self.coupling_layer[i]
        return thing
