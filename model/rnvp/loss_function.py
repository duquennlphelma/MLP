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

    def forward(self, z, det_J):
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        log_px = det_J + log_pz
        nll = -log_px
        bpd = nll * np.log2(np.exp(1)) / np.prod(z.shape[1:])
        return bpd.mean()
