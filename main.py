import torch
from model.rnvp.RNVP import RNVP
from data.data_simple import x_sample, batch


if __name__ == '__main__':



    # data
    point = torch.Tensor([2, 3])

    # lengths
    # from shape of a data point
    D = 2
    print(D)
    # ceiling of a half of D
    d = 1


    model = RNVP(D, d)
    # trying forward on a one 2d-point
    out = model.forward(point)
    #print(out)

    # preparation for using the loss function
    s_sum = 0
    for i in range(batch):
        # add all scaling factors s from all coupling layers
        s_sum += model.layer_0.s_recent + model.layer_1.s_recent

    # calculate output in a batch (todo: reformat it)
    # p_z is a real valued function defining standard normal distribution
    z_output = torch.zeros((batch, D))
    for i in range(batch):
        z_output[i] = model.forward(x_sample[i])
    # print(z_output)

    # next: calculate loss function value of this output here
    # further: train the network


