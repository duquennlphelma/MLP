import numpy as np
import torch

def loss_log(z, det_J):
    z_detach= z.torch.tensor.detach()
    N=len(z_detach)
    print(N)
    pz= -N/2 *np.log(2*np.pi) -1/2 * torch.sum(torch.mul(z_detach,z_detach))
    return (np.log(pz) + det_J)
