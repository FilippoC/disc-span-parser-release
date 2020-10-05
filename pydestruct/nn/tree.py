import torch
import torch.autograd
import torch.nn as nn


class MatrixTree(nn.Module):
    """

    Updateed code of: https://github.com/gujiuxiang/NMT.pytorch/blob/master/onmt/modules/StructuredAttention.py

    instead, this ones take an adjacency matrix
    """

    def __init__(self, eps=1e-5):
        self.eps = eps
        super(MatrixTree, self).__init__()

    def forward(self, input):
        exp_weights = input.exp() + self.eps
        output = input.clone()

        for b in range(input.size(0)):
            batch_exp_weights = exp_weights[b]
            batch_no_root = batch_exp_weights[1:, 1:]

            lap = batch_no_root.masked_fill(torch.autograd.Variable(torch.eye(batch_no_root.size(1), device=input.device).ne(0)), 0)
            lap = -lap + torch.diag(lap.sum(0))

            # store roots on diagonal
            lap[0] = batch_exp_weights[0, 1:]

            inv_laplacian = lap.inverse()

            factor = inv_laplacian.diag().unsqueeze(1).expand_as(batch_no_root).transpose(0, 1)
            term1 = batch_no_root.mul(factor).clone()
            term2 = batch_no_root.mul(inv_laplacian.transpose(0, 1)).clone()
            term1[:, 0] = 0
            term2[0] = 0
            roots_output = batch_exp_weights[0, 1:].mul(inv_laplacian.transpose(0, 1)[0])
            output[b, 1:, 1:] = term1 - term2
            output[b, 0, 1:] = roots_output
            output[b, :, 0] = 0
        return output


def log_partition(weights, eps=1e-5):
    """
    log paritition of the distribution over rooted spanning arborescence, with a single root.
    Input must be the weighted adjacency matrix.
    Batches are not supported.
    """
    exp_weights = weights.exp() + eps
    no_root = exp_weights[1:, 1:]

    lap = no_root.masked_fill(torch.autograd.Variable(torch.eye(no_root.size(0), device=weights.device).ne(0)), 0)
    lap = -lap + torch.diag(lap.sum(0))

    # add root weights
    lap[0] = exp_weights[0, 1:]
    inv_laplacian = lap.inverse()

    return -inv_laplacian.logdet()
