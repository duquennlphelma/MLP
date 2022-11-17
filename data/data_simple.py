import torch
# todo: change the sample to a non random image
batch = 100

x_sample = torch.randn(batch*2)
x_sample = x_sample.reshape(batch, 2)

