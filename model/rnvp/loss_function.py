import numpy as np
import torch


def loss_log(z, det_J):
    prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
    prior_ll = prior_ll.view(z.size(0), -1).sum(-1) \
               - np.log(2) * np.prod(z.size()[1:])
    ll = prior_ll + det_J
    nll = -ll.mean()
    """
    # print('z:\n', z)

    N = len(z)

    #multidimensional normal law
    pz = -N/2 * np.log(2*np.pi) - 1/2 * torch.sum(torch.mul(z, z))

    return -pz + torch.sum(det_J) """

    return nll

