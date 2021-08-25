import torch
import torch.nn as nn
import numpy as np
import sys
import disc_span_parser

"""
class MarginLoss(nn.Module):
    def __init__(self, complexity, ill_nested, reduction="mean"):
        super(MarginLoss, self).__init__()
        self.complexity = complexity
        self.ill_nested = ill_nested
        self.mean = (reduction == "mean")

    # assume that weights are given in a matrix of size (n words + 2, n words + 2)
    # so the first word will be ignored
    def forward(self, dep_weights, label_weights, sentences):
        # remove the first line/col that corresponds to BOS tag
        dep_weights = {k: v[:, 1:, 1:] for k, v in dep_weights.items()}
        label_weights = {k: v[:, 1:, 1:] for k, v in label_weights.items()}

        n_max_words = dep_weights["cont"].shape[1]  # it contains bos but they will always be masked
        n_batch = dep_weights["cont"].shape[0]
        device = dep_weights["cont"].device

        # compute gold indices and loss augmented weight
        loss_augmented_weights = dict()
        gold_indices = dict()
        for k in dep_weights.keys():
            n_labels = label_weights[k].shape[3]
            weights = dep_weights[k] + label_weights[k]
            # the last label will be used as a "fake empty label" value
            g = torch.empty((n_batch, n_max_words, n_max_words, n_labels), requires_grad=False, device=device).fill_(0.)
            for b in range(n_batch):
                x = sentences[b][k + "_spans"][0].to(device)
                y = sentences[b][k + "_spans"][1].to(device)
                z = sentences[b][k + "_labels"].to(device)
                g[b, x, y, z] = 1.

            gold_indices[k] = g
            loss_augmented_weights[k] = weights + (1. - g)

        # compute argmax
        pred_indices = {
            k: torch.empty((n_batch, n_max_words, n_max_words, label_weights[k].shape[3]), requires_grad=False, device=device).fill_(0.)
            for k in dep_weights.keys()
        }
        for b in range(n_batch):
            n_words = len(sentences[b]["words"]) - 2
            cont_spans = loss_augmented_weights["cont"][b, :n_words, :n_words]
            disc_spans = loss_augmented_weights["disc"][b, :n_words, :n_words]
            gap_spans = loss_augmented_weights["gap"][b, :n_words, :n_words]

            pred_cst = disc_span_parser.argmax_as_list_parallel(
                list([cont_spans]),
                list([disc_spans]),
                list([gap_spans]),
                None,
                self.complexity,
                self.ill_nested
            )[0]

            for label, i, k, l, j in pred_cst:
                if k < 0:
                    pred_indices["cont"][b, i, j, label] += 1.
                else:
                    pred_indices["disc"][b, i, j, label] += 1.
                    pred_indices["gap"][b, k+1, l-1, label] += 1.

        # try to normalize with  / gold_indices[k].sum()
        loss = [loss_augmented_weights[k] * (pred_indices[k] - gold_indices[k]) for k in dep_weights.keys()]
        loss = [l.sum() for l in loss]
        loss = sum(loss).sum()
        if self.mean:
            loss = loss / n_words
        return loss


class CorrectedBatchUnstructuredProbLoss(nn.Module):
    def __init__(self, joint=True, reduction="mean"):
        super(CorrectedBatchUnstructuredProbLoss, self).__init__()
        self.joint = joint
        self.builder = nn.BCEWithLogitsLoss(reduction=reduction)

    # assume that weights are given in a matrix of size (n words + 2, n words + 2)
    # so the first word will be ignored
    def forward(self, dep_weights, label_weights, sentences):
        # remove the first line/col that corresponds to BOS tag
        dep_weights = {k: v[:, 1:, 1:] for k, v in dep_weights.items()}
        label_weights = {k: v[:, 1:, 1:] for k, v in label_weights.items()}

        n_max_words = dep_weights["cont"].shape[1]  # it contains bos but they will always be maked
        n_batch = dep_weights["cont"].shape[0]
        device = dep_weights["cont"].device

        # build mask
        triangle = torch.ones((n_max_words, n_max_words), dtype=bool, device=device, requires_grad=False).triu_(diagonal=0)

        # dim: (n_batch, max len)
        sentence_sizes = torch.LongTensor([len(sentence["words"]) for sentence in sentences]).to(device)
        size_mask = torch.arange(n_max_words, device=device).unsqueeze(0) < sentence_sizes.unsqueeze(1)

        # our triangular mask!
        mask = triangle.expand(n_batch, -1, -1)
        mask = mask * size_mask.unsqueeze(2)
        mask = mask * size_mask.unsqueeze(1)

        if self.joint:
            loss = list()
            for k in dep_weights.keys():
                n_labels = label_weights[k].shape[3]
                weights = dep_weights[k] + label_weights[k]
                # the last label will be used as a "fake empty label" value
                gold_indices = torch.empty((n_batch, n_max_words, n_max_words, n_labels), requires_grad=False, device=device).fill_(0.)
                for b in range(n_batch):
                    x = sentences[b][k + "_spans"][0].to(device)
                    y = sentences[b][k + "_spans"][1].to(device)
                    z = sentences[b][k + "_labels"].to(device)
                    gold_indices[b, x, y, z] = 1.

                weights = weights[mask]
                gold_indices = gold_indices[mask]
                loss.append(self.builder(weights, gold_indices))
            return sum(loss)
        else:
            raise NotImplementedError()
"""

# this is the "correct" loss
class BatchUnstructuredApproximateProbLoss(nn.Module):
    def __init__(self, joint=True, reduction="mean"):
        super(BatchUnstructuredApproximateProbLoss, self).__init__()
        self.joint = joint
        self.builder = nn.CrossEntropyLoss(reduction=reduction)
        if not joint:
            self.binary_builder = nn.BCEWithLogitsLoss(reduction=reduction)

    # assume that weights are given in a matrix of size (n words + 2, n words + 2)
    # so the first word will be ignored
    def forward(self, dep_weights, label_weights, sentences):
        # remove the first line/col that corresponds to BOS tag
        dep_weights = {k: v[:, 1:, 1:] for k, v in dep_weights.items()}
        label_weights = {k: v[:, 1:, 1:] for k, v in label_weights.items()}

        n_max_words = dep_weights["cont"].shape[1]  # it contains bos but they will always be maked
        n_batch = dep_weights["cont"].shape[0]
        device = dep_weights["cont"].device

        # build mask
        triangle = torch.ones((n_max_words, n_max_words), dtype=bool, device=device, requires_grad=False).triu_(diagonal=0)

        # dim: (n_batch, max len)
        sentence_sizes = torch.LongTensor([len(sentence["words"]) for sentence in sentences]).to(device)
        size_mask = torch.arange(n_max_words, device=device).unsqueeze(0) < sentence_sizes.unsqueeze(1)

        # our triangular mask!
        mask = triangle.expand(n_batch, -1, -1)
        mask = mask * size_mask.unsqueeze(2)
        mask = mask * size_mask.unsqueeze(1)

        if self.joint:
            loss = list()
            for k in dep_weights.keys():
                n_labels = label_weights[k].shape[3]
                weights = dep_weights[k] + label_weights[k]
                # the last label will be used as a "fake empty label" value
                gold_indices = torch.empty((n_batch, n_max_words, n_max_words), dtype=torch.long, requires_grad=False, device=device).fill_(n_labels)
                for b in range(n_batch):
                    x = sentences[b][k + "_spans"][0].to(device)
                    y = sentences[b][k + "_spans"][1].to(device)
                    z = sentences[b][k + "_labels"].to(device)
                    gold_indices[b, x, y] = z

                weights = weights[mask]
                # "fake" label
                weights = torch.cat([weights, torch.zeros((weights.shape[0], 1), device=device)], dim=1)
                gold_indices = gold_indices[mask]
                loss.append(self.builder(weights, gold_indices))
            return sum(loss)
        else:
            loss = list()
            for k in dep_weights.keys():
                n_labels = label_weights[k].shape[3]

                xs = [sentences[b][k + "_spans"][0].to(device) for b in range(n_batch)]
                ys = [sentences[b][k + "_spans"][1].to(device) for b in range(n_batch)]
                # span loss
                gold_span_indices = torch.zeros((n_batch, n_max_words, n_max_words), dtype=torch.float).to(device)
                for b in range(n_batch):
                    gold_span_indices[b, xs[b], ys[b]] = 1.

                gold_span_indices = gold_span_indices[mask]
                loss.append(self.binary_builder(dep_weights[k].squeeze(-1)[mask], gold_span_indices))

                # label loss
                label_mask = torch.zeros((n_batch, n_max_words, n_max_words), device=device, requires_grad=False, dtype=torch.bool)
                for b in range(n_batch):
                    label_mask[b, xs[b], ys[b]] = True

                # this is inefficient, but I do this to ensure the same order after unmasking
                gold_label_indices = torch.zeros((n_batch, n_max_words, n_max_words), device=device, requires_grad=False, dtype=torch.long)
                for b in range(n_batch):
                    gold_label_indices[b, xs[b], ys[b]] = sentences[b][k + "_labels"].to(device)

                loss.append(self.builder(label_weights[k][mask], gold_label_indices[mask]))

            # maybe it should be this?
            # return sum(loss)
            return sum(l.sum() for l in loss)

# why is this one wrong?
"""
class BatchUnstructuredCorrectProbLoss(nn.Module):
    def __init__(self, joint=True, reduction="mean"):
        super(BatchUnstructuredCorrectProbLoss, self).__init__()
        if not joint:
            raise RuntimeError("Not implemented")
        self.builder = nn.CrossEntropyLoss(reduction=reduction)

    def build_indices(self, name, n_words, device):
        id_x = list()
        id_y = list()
        if name == "cont":
            for i in range(n_words):
                for j in range(i, n_words):
                    id_x.append(i)
                    id_y.append(j)

        elif name =="disc" or name == "gap":
            for i in range(n_words):
                for j in range(i, n_words):
                    for k in range(i + 1, j):
                        for l in range(k, j):
                            if name == "disc":
                                id_x.append(i)
                                id_y.append(j)
                            else:
                                id_x.append(k)
                                id_y.append(l)
        else:
            raise RuntimeError("Invalid name: %s" % name)

        return (
            torch.LongTensor(id_x).to(device),
            torch.LongTensor(id_y).to(device)
        )

    # assume that weights are given in a matrix of size (n words + 2, n words + 2)
    # so the first word will be ignored
    def forward(self, dep_weights, label_weights, sentences):
        # remove the first line/col that corresponds to BOS tag
        dep_weights = {k: v[:, 1:, 1:] for k, v in dep_weights.items()}
        label_weights = {k: v[:, 1:, 1:] for k, v in label_weights.items()}
        n_batch = dep_weights["cont"].shape[0]
        device = dep_weights["cont"].device

        loss = list()
        weights = {k: dep_weights[k] + label_weights[k] for k in dep_weights.keys()}
        for k in dep_weights.keys():
            n_labels = weights[k].shape[3]

            for b, sentence in enumerate(sentences):
                n_words = len(sentence["words"]) - 2
                id_x, id_y = self.build_indices(k, n_words, weights[k].device)


                # the last label will be used as a "fake empty label" value
                gold_indices = torch.empty((n_words, n_words), dtype=torch.long, requires_grad=False, device=device).fill_(n_labels)
                x = sentences[b][k + "_spans"][0].to(device)
                y = sentences[b][k + "_spans"][1].to(device)
                z = sentences[b][k + "_labels"].to(device)
                gold_indices[x, y] = z

                sentence_weights = weights[k][b, id_x, id_y]
                # "fake" label
                sentence_weights = torch.cat([sentence_weights, torch.zeros((sentence_weights.shape[0], 1), device=device)], dim=1)
                gold_indices = gold_indices[id_x, id_y]
                loss.append(self.builder(sentence_weights, gold_indices))
        return sum(loss)
"""