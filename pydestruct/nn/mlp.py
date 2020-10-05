
import torch.nn as nn
from pydestruct.nn.dropout import SharedDropout

class MLP(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0, activation="tanh", shared_dropout=True, negative_slope=0.1):
        super(MLP, self).__init__()


        if activation == "tanh":
            activation = nn.Tanh()
        elif activation == "relu":
            activation = nn.ReLU()
        elif activation == "elu":
            activation = nn.ELU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=negative_slope)
        else:
            raise RuntimeError("Unknown activation function: %s" % activation)

        self.seq = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            activation,
            SharedDropout(p=dropout) if shared_dropout else nn.Dropout(dropout)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.seq[0].weight)
        nn.init.zeros_(self.seq[0].bias)

    def forward(self, x):
        return self.seq(x)
