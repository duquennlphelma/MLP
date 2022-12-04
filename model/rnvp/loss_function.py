import numpy as np
import torch

def loss_log(z, det_J):
    N=z.size()
    pz= -N/2 *np.log(2*np.pi) -1/2 * torch.sum(torch.mul(z,z))
    return np.log(pz) + det_J
