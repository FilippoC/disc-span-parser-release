import torch
import torch.nn as nn
import torch.nn.functional as F
import pydestruct
import disc_span_parser
import cpp_disc_span_parser
import pydestruct.input


def get_indices(cst_weights, pred_cst, device="cpu"):
    pred_cst_indices = torch.zeros_like(cst_weights, requires_grad=False, device="cpu")
    for label, i, j in pred_cst:
        pred_cst_indices[i, j, label] += 1.
    pred_cst_indices = pred_cst_indices.to(device)
    return pred_cst_indices


class StructuredMarginLoss(nn.Module):
    def __init__(self, complexity, ill_nested):
        super(StructuredMarginLoss, self).__init__()
        self.complexity = complexity
        self.ill_nested = ill_nested

    def forward(self, cont_weights, disc_weights, gap_weights, sentences):
        pred_span_list = disc_span_parser.argmax_as_list_parallel(
            cont_weights,
            disc_weights,
            gap_weights,
            [sentence["torch"]["gold_spans"] for sentence in sentences],
            self.complexity,
            self.ill_nested
        )

        losses = list()
        for i, pred_tree in enumerate(pred_span_list):
            sentence = sentences[i]
            torch_inputs = sentence["torch"]
            # compute constituent loss
            pred_cont_indices = get_indices(
                    cont_weights[i],
                    ((label, i, j) for label, i, k, l, j in pred_tree if k < 0),
                    device=cont_weights[i].device
            )
            pred_disc_indices = get_indices(
                disc_weights[i],
                ((label, i, j) for label, i, k, l, j in pred_tree if k >= 0),
                device=disc_weights[i].device
            )
            pred_gap_indices = get_indices(
                gap_weights[i],
                ((label, k+1, l-1) for label, i, k, l, j in pred_tree if k >= 0),
                device=gap_weights[i].device
            )

            loss = torch.sum(
                (pred_cont_indices - torch_inputs["cst_gold_cont"].to(pred_cont_indices.device))
                * cont_weights[i]
            )
            loss += torch.sum(
                (pred_disc_indices - torch_inputs["cst_gold_disc"].to(pred_disc_indices.device))
                * disc_weights[i]
            )
            loss += torch.sum(
                (pred_gap_indices - torch_inputs["cst_gold_gap"].to(pred_gap_indices.device))
                * gap_weights[i]
            )
            loss += len(set(pred_tree) - sentence["constituents_indices"])

            losses.append(loss)

        return losses


class UnstructuredProbLoss(nn.Module):
    def __init__(self, reduction="sum"):
        super(UnstructuredProbLoss, self).__init__()
        if reduction != "sum":
            raise NotImplementedError()

    def forward(self, cont_weights, disc_weights, gap_weights, sentences):
        losses = list()
        for sid in range(len(cont_weights)):
            n_words = disc_weights[sid].shape[0]
            n_cont_labels = cont_weights[sid].shape[2]
            n_disc_labels = disc_weights[sid].shape[2]

            # add fake label
            null_label_tensor = torch.zeros((n_words, n_words, 1), device=cont_weights[sid].device)
            current_cont_weights = torch.cat([cont_weights[sid], null_label_tensor], dim=2)
            current_disc_weights = torch.cat([disc_weights[sid], null_label_tensor], dim=2)
            current_gap_weights = torch.cat([gap_weights[sid], null_label_tensor], dim=2)

            # continuous constituents loss

            cont_gold_labels = n_cont_labels * torch.ones((n_words, n_words), dtype=torch.long, device=current_cont_weights.device)
            for label, i, k, l, j in sentences[sid]["constituents_indices"]:
                if k < 0:
                    cont_gold_labels[i, j] = label
            ones = torch.ones((n_words, n_words), device=current_cont_weights.device)
            triangular0 = torch.triu(ones, diagonal=0)
            cont_loss_mask = triangular0

            loss = (F.cross_entropy(
                        current_cont_weights.reshape(-1, n_cont_labels + 1),
                        cont_gold_labels.reshape(-1),
                        reduction="none"
                    ) * cont_loss_mask.reshape(-1)).sum()

            # discontinuous constituents loss

            disc_gold_cst = dict()
            for label, i, k, l, j in sentences[sid]["constituents_indices"]:
                if k >= 0:
                    disc_gold_cst[(i, k, l, j)] = label

            cont_indices = list()
            disc_indices = list()
            label_indices = list()
            for i in range(n_words):
                for k in range(i, n_words):
                    for l in range(k + 2, n_words):
                        for j in range(l, n_words):
                            gold_label = disc_gold_cst.get((i, k, l, j), n_disc_labels)
                            label_indices.append(gold_label)

                            cont_indices.append(i * n_words + j)
                            disc_indices.append((k + 1) * n_words + (l - 1))

            if len(label_indices) > 0:
                weights = current_disc_weights.reshape(-1, n_disc_labels + 1)[cont_indices] \
                            + current_gap_weights.reshape(-1, n_disc_labels + 1)[disc_indices]
                loss = loss + F.cross_entropy(
                        weights,
                        torch.LongTensor(label_indices).to(current_disc_weights.device),
                        reduction="sum"
                    )

            losses.append(loss)
        return losses

    """
    Too slow...
    def forward(self, cont_weights, disc_weights, gap_weights, sentences):
        losses = list()
        for i in range(len(cont_weights)):
            n_words = disc_weights[i].shape[0]
            n_cont_labels = cont_weights[i].shape[2]
            n_disc_labels = disc_weights[i].shape[2]

            null_label_tensor = torch.zeros((n_words, n_words, 1), device=cont_weights[i].device)
            current_cont_weights = torch.cat([cont_weights[i], null_label_tensor], dim=2)

            current_disc_weights = torch.cat([disc_weights[i], null_label_tensor], dim=2)
            current_gap_weights = torch.cat([gap_weights[i], null_label_tensor], dim=2)
            combined_dist_weights = current_disc_weights.reshape(n_words, 1, 1, n_words, n_disc_labels+1) + current_gap_weights.reshape(1, n_words, n_words, 1, n_disc_labels+1)

            cont_gold_labels = n_cont_labels * torch.ones((n_words, n_words), dtype=torch.long, device=current_cont_weights.device)
            disc_gold_labels = n_disc_labels * torch.ones((n_words, n_words, n_words, n_words), dtype=torch.long, device=current_cont_weights.device)

            for label, i, k, l, j in sentences[i]["constituents_indices"]:
                if k >= 0:
                    disc_gold_labels[i, k+1, l - 1, j] = label
                else:
                    cont_gold_labels[i, j] = label

            ones = torch.ones((n_words, n_words), device=current_cont_weights.device)
            triangular0 = torch.triu(ones, diagonal=0)
            cont_loss_mask = triangular0
            #triangular2 = torch.triu(ones, diagonal=2)
            #disc_loss_mask = triangular0.reshape((n_words, n_words, 1, 1)) * triangular2.reshape((1, n_words, n_words, 1)) * triangular0.reshape((1, 1, n_words, n_words))

            triangular1 = torch.triu(ones, diagonal=1)
            disc_loss_mask = triangular1.reshape((n_words, n_words, 1, 1)) * triangular0.reshape((1, n_words, n_words, 1)) * triangular1.reshape((1, 1, n_words, n_words))

            loss = (F.cross_entropy(
                        current_cont_weights.reshape(-1, n_cont_labels + 1),
                        cont_gold_labels.reshape(-1),
                        reduction="none"
                    ) * cont_loss_mask.reshape(-1)).sum() + \
                   (F.cross_entropy(
                        combined_dist_weights.reshape(-1, n_disc_labels + 1),
                        disc_gold_labels.reshape(-1),
                         reduction="none"
                    ) * disc_loss_mask.reshape(-1)).sum()

            losses.append(loss)
        return losses
"""
