import torch

# todo: change the sample to a non random image
# generate it through applying a transformation to the random smaple

# global variables
D = 2  # nr of dimensions (for an image its 32^2 or 64^2)
batch = 100

x_sample = torch.randn(batch*D)
x_sample = x_sample.reshape(batch, D)

