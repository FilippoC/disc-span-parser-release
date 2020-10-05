import torch
import pydestruct.network

def constituents_as_matrix(size, n_constituents, constituents, device="cpu"):
    indices = torch.zeros((size, size, n_constituents), requires_grad=False, device="cpu")
    for label, i, j in constituents:
        # we use +=1 so it also works for discontinuous constituents
        indices[i, j, label] += 1
    return indices.to(device=device)


def dependencies_as_matrix(dependencies, device="cpu"):
    size = len(dependencies) + 1
    dep_indices = torch.zeros((size, size), requires_grad=False, device="cpu")
    for mod, head in enumerate(dependencies):
        dep_indices[head + 1, mod + 1] = 1.
    return dep_indices.to(device)


def tensor_from_dict(dict, data, device="cpu", out_of_range_unk=False):
    if out_of_range_unk:
        ids = [dict._word_to_id.get(item.lower() if dict._lower else item, len(dict._word_to_id)) for item in data]
    else:
        ids = [dict.word_to_id(item) for item in data]
    if dict.has_boundaries():
        ids = [dict.bos_id()] + ids + [dict.eos_id()]
    return torch.tensor(ids, dtype=torch.long, device=device, requires_grad=False)


def build(dicts, data, allow_unk_labels=False, elmo=False, bert_tokenizer=None, device="cpu"):
    # check if constituents are continuous or not
    # this is required beforehand because a sentence can have 0 constituents
    # however this test will fail if ALL sentences in data have 0 constituents
    # but this is probably ok?
    continuous_constituents = True
    if "constituents" in data[0]:
        for sentence in data:
            if len(sentence["constituents"]) > 0:
                if len(next(iter(sentence["constituents"]))) == 5:
                    continuous_constituents = False
                break
        else:
            raise RuntimeError("All sentences have 0 constituents, so this test fail... :( ")

    for sentence in data:
        torch_inputs = {}
        n_words = len(sentence["words"])

        torch_inputs["words"] = tensor_from_dict(dicts["words"], sentence["words"], device)
        torch_inputs["chars"] = [tensor_from_dict(dicts["chars"], word, device) for word in sentence["words"]]

        if elmo:
            # I am to lazy too understand how to build that here
            torch_inputs["elmo"] = [pydestruct.network.ELMO_TOKEN_MAPPING.get(word, word) for word in sentence["words"]]

        if bert_tokenizer is not None:
            torch_inputs["bert"] = bert_tokenizer(sentence["words"])

        if "tags" in sentence:
            torch_inputs["tags"] = tensor_from_dict(dicts["tags"], sentence["tags"], device)

        if "heads" in sentence:
            deps = torch.zeros((n_words + 1, n_words + 1), device=device, requires_grad=False)
            for mod, head in enumerate(sentence["heads"]):
                deps[head + 1, mod + 1] = 1.
            torch_inputs["dependencies"] = deps
            torch_inputs["dependency_penalties"] = 1 - deps

        if "constituents" in sentence:
            c = set()
            for cst in sentence["constituents"]:
                label = cst[0]
                try:
                    if len(cst) == 3 or cst[2] < 0:
                        label_id = dicts["labels"].word_to_id(label)
                    elif len(cst) == 5:
                        label_id = dicts["disc_labels"].word_to_id(label)
                    else:
                        raise RuntimeError("Weird constituent: ", cst)
                except KeyError as e:
                    if allow_unk_labels:
                        label_id = -1
                    else:
                        raise e
                # in the weight matrix, the id is the last element
                c.add((label_id,) + cst[1:])
            sentence["constituents_indices"] = c

            if continuous_constituents:
                torch_inputs["cst_gold"] = constituents_as_matrix(n_words, len(dicts["labels"]), sentence["constituents_indices"], device=device)
                torch_inputs["cst_penalties"] = 1. - torch_inputs["cst_gold"]
            else:
                cont = [(label, i, j) for label, i, k, l, j in sentence["constituents_indices"] if k < 0]
                disc = [(label, i, j) for label, i, k, l, j in sentence["constituents_indices"] if k >= 0]
                gap = [(label, k + 1, l - 1) for label, i, k, l, j in sentence["constituents_indices"] if k >= 0]
                torch_inputs["cst_gold_cont"] = constituents_as_matrix(n_words, len(dicts["labels"]), cont, device=device)
                torch_inputs["cst_gold_disc"] = constituents_as_matrix(n_words, len(dicts["disc_labels"]), disc, device=device)
                torch_inputs["cst_gold_gap"] = constituents_as_matrix(n_words, len(dicts["disc_labels"]), gap, device=device)

        if "all_unlabeled_constituents" in sentence:
            ucst = torch.ones((n_words, n_words), device=device, requires_grad=False)
            for i, j in sentence["all_unlabeled_constituents"]:
                ucst[i, j] = 0.
            ucst[ucst > 0] = float("-inf")
            torch_inputs["all_unlabeled_constituents"] = ucst

        sentence["torch"] = torch_inputs
