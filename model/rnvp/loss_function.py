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
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, z, det_J):
        print('------enter loss------')
        print('print size z ', y.size())
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        print('print size log pz: ', log_pz.size())
        log_px = det_J + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(z.shape[1:])
        return bpd.mean()

    """
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) - np.log(2) * np.prod(z.size()[1:])
        ll = prior_ll + det_J
        nll = -ll.mean()

        #log_pz = self.prior.log_prob(z)
        #return (-log_pz + det_J).mean()
        return nll

    """



