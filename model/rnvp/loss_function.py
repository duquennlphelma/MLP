import numpy as np
import torch
from torch import nn


class NLL(nn.Module):
    """
    Negative log likelihood
    """
    def __init__(self):
        super(NLL, self).__init__()
        #self.prior = torch.distributions.MultivariateNormal(torch.zeros((1,28,28)), torch.eye(28))

    def forward(self, z, det_J):

        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) - np.log(2) * np.prod(z.size()[1:])
        ll = prior_ll + det_J
        nll = -ll.mean()

        #log_pz = self.prior.log_prob(z)
        #return (-log_pz + det_J).mean()
        return nll



