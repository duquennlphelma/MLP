import numpy as np
import torch

def loss_log(z, det_J):
    print('z:\n', z)
    z_detach = z.detach()
    print('z_detach:\n', z_detach)
    N = len(z_detach)
    print('len(z_detach)\n', N)

    #multidimensional normal law
    pz = -N/2 * np.log(2*np.pi) - 1/2 * torch.sum(torch.mul(z_detach,z_detach))
    print("torch.mul chelou", torch.mul(z_detach,z_detach))
    print("len pz and pz \n", pz)
    #are we really summing over the batch doing torch.sum(det_J) ? yes we are
    print("which size is det_J?\n", len(det_J))
    return pz - torch.sum(det_J)
