import torch
import torch.nn as nn

# as of now this ia very simple, but it could be extended to allow for an hidden layer!

class TaggerModule(nn.Module):
    def __init__(self, input_dim, n_labels, bias=True):
        super(TaggerModule, self).__init__()
        self.linear = nn.Linear(input_dim, n_labels, bias=bias)

    def forward(self, input):
        return self.linear(input)