import tensorboardX

_tbx_writer = None


def open(path):
    global _tbx_writer
    _tbx_writer = tensorboardX.SummaryWriter(path)


def add_scalar(name, value, it):
    global _tbx_writer
    if _tbx_writer is not None:
        _tbx_writer.add_scalar(name, value, it)