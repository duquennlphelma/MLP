import numpy as np
import torch


def loss_log(z, det_J):

    # print('z:\n', z)

    N = len(z)

    #multidimensional normal law
    pz = -N/2 * np.log(2*np.pi) - 1/2 * torch.sum(torch.mul(z, z))

    return -pz + torch.sum(det_J)
