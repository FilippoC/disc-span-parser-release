import ctypes
import torch.autograd
import pydestruct

# TODO: this use the old C++ API base on ctype!
# so it won't work...
# WARNING:
# you should not use directly the functions/classes of this file
# please use what is in the __init__.py file instead


def _log_partition_forward(ctx, input, fencepost):
    pydestruct._check_labeled(input)
    size = input.size()[0]

    marginals = torch.zeros_like(input)
    log_z = pydestruct.CPP_LIB.log_partition_and_marginals(
        pydestruct.IO_CHART,
        size,
        input.size()[2],
        ctypes.c_void_p(input.data_ptr()),
        ctypes.c_void_p(marginals.data_ptr())
    )
    ctx.save_for_backward(marginals)

    return torch.tensor(log_z)


def _log_partition_lex_forward(ctx, cst_input, dep_input):
    pydestruct._check_labeled(cst_input)
    pydestruct._check_unlabeled(dep_input)
    size = cst_input.size()[0]

    cst_marginals = torch.zeros_like(cst_input)
    dep_marginals = torch.zeros_like(dep_input)
    log_z = pydestruct.CPP_LIB.log_partition_and_marginals_lex(
        pydestruct.IO_LEX_CHART,
        size,
        cst_input.size()[2],
        ctypes.c_void_p(cst_input.data_ptr()),
        ctypes.c_void_p(dep_input.data_ptr()),
        ctypes.c_void_p(cst_marginals.data_ptr()),
        ctypes.c_void_p(dep_marginals.data_ptr())
    )
    ctx.save_for_backward(cst_marginals, dep_marginals)

    return torch.tensor(log_z)


def _log_partition_disc(ctx, cont, disc, gap):
    pydestruct._check_labeled(cont)
    pydestruct._check_labeled(disc)
    pydestruct._check_labeled(gap)
    size = cont.size()[0]

    cont_marginals = torch.zeros_like(cont)
    disc_marginals = torch.zeros_like(disc)
    gap_marginals = torch.zeros_like(gap)
    log_z = pydestruct.CPP_LIB.log_partition_and_marginals_disc(
        pydestruct.IO_DISC_CHART,
        size,
        cont.size()[2],
        ctypes.c_void_p(cont.data_ptr()),
        ctypes.c_void_p(disc.data_ptr()),
        ctypes.c_void_p(gap.data_ptr()),
        ctypes.c_void_p(cont_marginals.data_ptr()),
        ctypes.c_void_p(disc_marginals.data_ptr()),
        ctypes.c_void_p(gap_marginals.data_ptr())
    )
    ctx.save_for_backward(cont_marginals, disc_marginals, gap_marginals)

    return torch.tensor(log_z)


class LogPartition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return _log_partition_forward(ctx, input, False)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.saved_tensors


class LogPartitionLex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cst_input, dep_input):
        return _log_partition_lex_forward(ctx, cst_input, dep_input, False)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.saved_tensors


class LogPartitionDisc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cont, disc, gap):
        return _log_partition_disc(ctx, cont, disc, gap, False)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.saved_tensors


class LogPartitionDiscFencepost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cont, disc, gap):
        return _log_partition_disc(ctx, cont, disc, gap, True)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.saved_tensors


log_partition = LogPartition.apply
log_partition_lex = LogPartitionLex.apply
log_partition_disc = LogPartitionDisc.apply
