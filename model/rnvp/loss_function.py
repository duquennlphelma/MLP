import numpy as np
import torch
from torch import nn


class NLL(nn.Module):
    """
    Negative log likelihood
    """
    def __init__(self):
        super(NLL, self).__init__()
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.prior2D = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    def forward(self, z, det_J):
        print('log_pz_size:', self.prior.log_prob(z).size())
        print('z_size:', z.size())

        if len(z.size())>2:
            log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])

            print('log_pz_sizesum:', log_pz.size())
            log_px = det_J + log_pz
            nll = -log_px
            bpd = nll * np.log2(np.exp(1)) / np.prod(z.shape[1:])
            return bpd.mean()

        else :
            log_pz = self.prior2D.log_prob(z)
            return (-log_pz + det_J).mean()

