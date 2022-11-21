import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, n_input_channel, n_output_channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channel, n_output_channel, kernel_size=3)
        self.norm_in=nn.BatchNorm2d(n_input_channel)
        self.norm_out=nn.BatchNorm2d(n_output_channel)

    def forward(self, x):
        x_copy=x.copy()
        x=self.norm_in(x)
        x=F.relu(x)
        x=self.conv1(x)
        x=self.norm_out(x)

        x = x+ x_copy

        return x