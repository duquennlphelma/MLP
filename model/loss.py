import torch
def loss(z_output, s_sum):
    return - torch.tensordot(z_output,z_output, dims=([-1], [-1])) - s_sum
