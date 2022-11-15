import numpy as np

# todo: throw those into the main channel
# todo: import real models not those scams
from model import s_forward_list
from model import model_forward
from data_simple import x_sample
from main import D


batch = 100  # nr of samples
coupling_layers_count = 2

# todo: s_forward will be a vector valued function, that takes d numbers
#     and gives D-d values
#     it will be a deep nn
#     then we will need to sum those vectors

s = 0
for i in range(batch):
    for l in range(coupling_layers_count):
        s += s_forward_list[l].T @ x_sample[i, :]

# todo: z_output will be computed from samples
# p_z is a standard normal
z_output = np.zeros((batch, D))
z_output = x_sample


def loss(x_sample, z_output, s):
    sum_of_probabilities_z = 0
    for i in range(batch):
        sum_of_probabilities_z += 1 / 2 / np.pi * np.exp(-z_output[i, :].T @ z_output[i, :] / 2)
    return - sum_of_probabilities_z - s
