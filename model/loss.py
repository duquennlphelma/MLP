import numpy as np

# todo: avoid numpy or change it to torch

def loss(x_input, z_output, s_sum):
    sum_of_probabilities_of_z = 0
    for i in range(x_input.shape[0]):
        sum_of_probabilities_of_z += 1 / 2 / np.pi * np.exp(-z_output[i, :].T @ z_output[i, :] / 2)
    return - sum_of_probabilities_of_z - s_sum
