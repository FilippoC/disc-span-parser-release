import torch
import torch.nn as nn

# Stolen from: https://github.com/yzhangcs/biaffine-parser/blob/master/parser/modules/dropout.py

class SharedDropout(nn.Module):
    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()
        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        if self.training and self.p > 0.:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask


class IndependentDropout(nn.Module):
    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()
        self.p = p

    def forward(self, *items):
        if self.training and self.p > 0.:
            # with -1 it should work for any input dim
            #masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
            #        for x in items]
            masks = [x.new_empty(x.shape[:-1]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]
        return items
