import torch
import torch.nn as nn
import math
import sys

from pydestruct.nn.mlp import MLP
from pydestruct.nn.dropout import SharedDropout
from pydestruct.nn.transformer import LayerNorm

class StandardDependencyModule(nn.Module):
    def __init__(self,
                 input_dim,
                 proj_dim,
                 n_labels,
                 output_bias=True,
                 activation="tanh",
                 dropout=0.
    ):
        super(StandardDependencyModule, self).__init__()

        self.activation_name = activation
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "elu":
            self.activation = torch.nn.ELU()
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            raise RuntimeError("Unknown activation function: %s" % self.activation_name)

        self.proj_dropout = nn.Dropout(dropout)
        self.head_projection = torch.nn.Parameter(data=torch.Tensor(input_dim, proj_dim))
        self.mod_projection = torch.nn.Parameter(data=torch.Tensor(input_dim, proj_dim))
        self.bias_projection = torch.nn.Parameter(data=torch.Tensor(1, proj_dim))

        self.label_output_projection = torch.nn.Parameter(data=torch.Tensor(proj_dim, n_labels))
        if output_bias:
            self.label_output_projection_bias = torch.nn.Parameter(data=torch.Tensor(1, n_labels))
        else:
            self.label_output_projection_bias = None

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            # using the default xavier init function is incorrect.
            # indeed, we split the W matrix into 2 parts in order to reduce computation
            # (trick from the papger of Kiperwasser and Goldberg)
            # the pytorch function will compute the values with the fan_in / 2 instead of the real fan_in
            # torch.nn.init.xavier_uniform_(self.head_projection)
            # torch.nn.init.xavier_uniform_(self.mod_projection)
            fan_in = self.head_projection.size()[1] * 2
            fan_out = self.head_projection.size()[2]
            if self.activation_name == "tanh":
                std = 1.0 * math.sqrt(2.0 / float(fan_in + fan_out))
                a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
                torch.nn.init.uniform_(self.head_projection, -a, a)
                torch.nn.init.uniform_(self.mod_projection, -a, a)
            else:
                a = 0
                activation = "relu" if self.activation_name == "elu" else self.activation_name
                gain = torch.nn.init.calculate_gain(activation, a)
                std = gain / math.sqrt(fan_in)
                bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
                self.head_projection.uniform_(-bound, bound)
                self.mod_projection.uniform_(-bound, bound)
            self.bias_projection.fill_(0.0)

            torch.nn.init.xavier_uniform_(self.label_output_projection)

            if self.label_output_projection_bias is not None:
                self.label_output_projection_bias.fill_(0.0)

    def forward(self, input):
        if len(input.size()) != 2:
            raise RuntimeError("Dependency nn: input must be a matrix: size x features")
        n_words = input.size()[0]

        # head_proj, mod_proj: (n_words, proj_dim)
        head_proj = input.matmul(self.head_projection)
        mod_proj = input.matmul(self.mod_projection)

        # change the dimension so we can rely on implicit broadcasting
        head_proj = head_proj.view(n_words, 1, -1)
        mod_proj = mod_proj.view(1, n_words, -1)

        # values: (n_words, n_words, proj_dim)
        values = head_proj + mod_proj # broadcast dimensions

        # now we apply a MLP to every couple of inputs, so we first need to redim
        # values: (n_words * n_words, proj_dim)
        values = values.view(n_words * n_words, -1)
        values = values + self.bias_projection
        values = self.activation(values)
        values = self.proj_dropout(values)

        # we apply the output projection
        # output_projection: (proj_dim, n_labels)
        # output: (n_words * n_words, n_labels)
        output = values[0].matmul(self.label_output_projection)
        if self.label_output_projection_bias is not None:
            output = output + self.label_output_projection_bias

        # we make the output a tensor
        # output: (n_words, n_words, n_labels)
        output = output.view(n_words, n_words, -1)

        return output


class BiaffineDependencyBase(nn.Module):
    def __init__(
            self,
            input_dim,
            proj_dim,
            n_labels,
            output_bias=True,
            mod_interaction=True,
            activation="tanh",
            dropout=0.,
            negative_slope=0.1 # for leaky relu
    ):
        super(BiaffineDependencyBase, self).__init__()

        if n_labels == 0:
            raise RuntimeError("Cannot create a module with 0 labels")

        self.head_projection = MLP(input_dim, proj_dim, activation=activation, dropout=dropout, negative_slope=negative_slope)
        self.mod_projection = MLP(input_dim, proj_dim, activation=activation, dropout=dropout, negative_slope=negative_slope)

        self.hm_labeled_interaction_tensor = torch.nn.Parameter(data=torch.Tensor(n_labels, proj_dim, proj_dim))
        self.hm_labeled_interaction_head = torch.nn.Parameter(data=torch.zeros(proj_dim, n_labels))

        if mod_interaction:
            self.hm_labeled_interaction_mod = torch.nn.Parameter(data=torch.zeros(proj_dim, n_labels))
        else:
            self.hm_labeled_interaction_mod = None

        if output_bias:
            self.output_bias = torch.nn.Parameter(data=torch.zeros(1, 1, n_labels))

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            torch.nn.init.zeros_(self.hm_labeled_interaction_tensor)
            torch.nn.init.zeros_(self.hm_labeled_interaction_head)
            if self.hm_labeled_interaction_mod is not None:
                torch.nn.init.zeros_(self.hm_labeled_interaction_mod)
            if self.output_bias is not None:
                torch.nn.init.zeros_(self.output_bias)

    def forward(self, input):
        #if len(input.size()) != 2:
        #    raise RuntimeError("Dependency nn: input must be a matrix: size x features")

        # head_proj, mod_proj: (n_words, proj_dim)
        head_proj = self.head_projection(input)
        mod_proj = self.mod_projection(input)

        values = (mod_proj @  (self.hm_labeled_interaction_tensor @ head_proj.transpose(0, 1))).transpose(0, -1)
        values = values + (head_proj @ self.hm_labeled_interaction_head).unsqueeze(1)

        if self.hm_labeled_interaction_mod is not None:
            values = values + (mod_proj @ self.hm_labeled_interaction_mod).unsqueeze(0)

        if self.output_bias is not None:
            values = values + self.output_bias

        return values

class BatchedMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 proj_dim,
                 n_labels,
                 output_bias=True,
                 activation="tanh",
                 dropout=0.):
        super().__init__()

        self.activation_name = activation
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "elu":
            self.activation = torch.nn.ELU()
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            raise RuntimeError("Unknown activation function: %s" % self.activation_name)

        self.seq = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, n_labels, bias=output_bias),
        )


    def forward(self, input):
        #input: (batch, word, features)
        x = input.unsqueeze(1)
        y = input.unsqueeze(2)
        return self.seq(x + y)


# stolen from https://github.com/yzhangcs/biaffine-parser/tree/master/parser
class BatchedBiaffine(nn.Module):
    def __init__(self, input_dim, proj_dim, n_labels=1, output_bias=True, activation="tanh", bias_x=True, bias_y=True, dropout=0, negative_slope=0.1):
        super(BatchedBiaffine, self).__init__()

        self.head_projection = MLP(input_dim, proj_dim, activation=activation, dropout=dropout, negative_slope=negative_slope)
        self.mod_projection = MLP(input_dim, proj_dim, activation=activation, dropout=dropout, negative_slope=negative_slope)

        self.n_in = input_dim
        self.n_out = n_labels
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_labels, proj_dim + bias_x, proj_dim + bias_y))

        if output_bias:
            self.output_bias = nn.Parameter(torch.Tensor(1, 1, 1, n_labels))
        else:
            self.output_bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.output_bias is not None:
            nn.init.zeros_(self.output_bias)

    def forward(self, features):
        x = self.head_projection(features)
        y = self.mod_projection(features)

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, seq_len, seq_len, n_out]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.permute(0, 2, 3, 1)

        if self.output_bias is not None:
            s = s + self.output_bias

        return s


class DependencyModule(nn.Module):
    def __init__(
            self,
             input_dim,
             proj_dim,
             n_labels,
             model="standard",
             fake_root=False,
             output_bias=True,
             span_bias=False,
             activation="tanh",
             mod_interaction=True, # usefull only for the biaffine model
             erase_diagonal=False,
             erase_root=False,
             dropout=0.
    ):
        super(DependencyModule, self).__init__()

        if n_labels == 0:
            raise RuntimeError("Cannot create a module with 0 labels")

        if model == "standard":
            model_class = StandardDependencyModule
        elif model == "biaffine":
            model_class = BiaffineDependencyBase
        else:
            raise RuntimeError("Unknown model: %s" % model)

        self.label_module = model_class(
            input_dim,
            proj_dim,
            n_labels,
            output_bias=output_bias,
            dropout=dropout,
            activation=activation
        )
        if span_bias:
            biaff_args = {"mod_interaction": mod_interaction} if model == "biaffine" else {}
            self.span_module = model_class(
                input_dim,
                proj_dim,
                1,
                output_bias=output_bias,
                dropout=dropout,
                activation=activation,
                **biaff_args
            )
        else:
            self.span_module = None

        self.erase_diagonal = erase_diagonal
        self.erase_root = erase_root

        if fake_root:
            self.fake_root = torch.nn.Parameter(torch.zeros(1, input_dim))
        else:
            self.fake_root = None

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            # Weird initialization
            # a good solution would to make some fake forward pass in the network
            # to compute the feature distributions at the input of this nn
            # But I am way too lazy to implement something like that
            if self.fake_root is not None:
                nn.init.uniform_(self.fake_root.data, -0.01, 0.01)

    def forward(self, input):
        if len(input.size()) != 2:
            raise RuntimeError("Dependency nn: input must be a matrix: size x features")
        n_words = input.size()[0]

        # add a fake root, useful for dependency parsing
        if self.fake_root is not None:
            input = torch.cat([self.fake_root, input], dim=0)
            n_words += 1

        output = self.label_module(input)
        if self.span_module is not None:
            output = output + self.span_module(input)

        if self.erase_diagonal:
            output = output + torch.diag(float("-inf") * torch.ones((n_words,), device=output.device)).unsqueeze(2)
        if self.erase_root:
            # set to -inf all dependencies (x, 0, label)
            infs = torch.zeros((1, n_words, 1), device=output.device)
            # TODO: check this
            infs[0, 0, 0] = float("-inf")
            output = output + infs

        return output

