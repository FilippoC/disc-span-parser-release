import torch
import collections
from pydestruct.dict import Dict
from pydestruct.input import tensor_from_dict
from pydestruct.text import is_punct
import pydestruct.nn.bert

def build_constituent_inputs(sentence, dicts, device="cpu"):
    cont_span1, cont_span2, cont_labels = list(), list(), list()
    disc_span1, disc_span2, disc_labels = list(), list(), list()
    gap_span1, gap_span2, gap_labels = list(), list(), list()
    for label, i, k, l, j in sentence["constituents"]:
        if k < 0:
            # continuous constituent
            cont_span1.append(i)
            cont_span2.append(j)
            cont_labels.append(dicts["cont_labels"].word_to_id(label))
        else:
            # discontinuous constituent
            disc_span1.append(i)
            disc_span2.append(j)
            disc_labels.append(dicts["disc_labels"].word_to_id(label))
            gap_span1.append(k+1)
            gap_span2.append(l-1)
            gap_labels.append(dicts["disc_labels"].word_to_id(label))

    return {
        "cont_spans": (
            torch.LongTensor(cont_span1, device=device),
            torch.LongTensor(cont_span2, device=device)
        ),
        "cont_labels": torch.LongTensor(cont_labels, device=device),
        "disc_spans": (
            torch.LongTensor(disc_span1, device=device),
            torch.LongTensor(disc_span2, device=device)
        ),
        "disc_labels": torch.LongTensor(disc_labels, device=device),
        "gap_spans": (
            torch.LongTensor(gap_span1, device=device),
            torch.LongTensor(gap_span2, device=device)
        ),
        "gap_labels": torch.LongTensor(gap_labels, device=device),
    }


def build_dictionnaries(data, char_boundaries=False, postag="tags", min_word_freq=1, external=None):
    dict_cont_labels = set()
    dict_disc_labels = set()
    dict_chars = set()
    dict_tags = set()
    dict_words = collections.defaultdict(lambda: 0)

    for sentence in data:
        dict_tags.update(sentence["tags"])
        for t in sentence["words"]:
            dict_words[t.lower()] += 1
            dict_chars.update(t)
        for cst in sentence["constituents" ]:
            if cst[2] < 0:
                dict_cont_labels.add(cst[0])
            else:
                dict_disc_labels.add(cst[0])

    dict_chars.add("**BOS_CHAR**")
    dict_chars.add("**EOS_CHAR**")
    dict_words = set(w for w, i in dict_words.items() if i >= min_word_freq)
    dict_words = Dict(dict_words, unk="#UNK#", boundaries=True, pad=True, lower=True)
    dict_chars = Dict(dict_chars, boundaries=char_boundaries)
    dict_cont_labels = Dict(dict_cont_labels)
    dict_disc_labels = Dict(dict_disc_labels)
    dict_tags = Dict(dict_tags, boundaries=True)
    ret = {
        "tags": dict_tags,
        "chars": dict_chars,
        "words": dict_words,
        "cont_labels":dict_cont_labels,
        "disc_labels": dict_disc_labels
    }

    if external is not None:
        external_words = set()
        with open(external, 'r') as f:
            for line in f:
                line = line.split()
                external_words.add(line[0])

        ret["ext_words"] = Dict(external_words, unk="#UNK#", boundaries=True, lower=True)

    return ret


def build_torch_input(sentence, dictionnaries, device="cpu", max_word_len=-1, copy_constituents=False, constituent_input=True, bert_tokenizer=None):
    ret = {
        "words": tensor_from_dict(dictionnaries["words"], [t for t in sentence["words"]], device=device),
        "chars":
            [torch.tensor([dictionnaries["chars"].word_to_id("**BOS_CHAR**")], dtype=torch.long, device=device, requires_grad=False)]
            +
            [
                tensor_from_dict(dictionnaries["chars"], t[:max_word_len] if max_word_len > 0 else t, device=device)
                for t in sentence["words"]
            ]
            +
            [torch.tensor([dictionnaries["chars"].word_to_id("**EOS_CHAR**")], dtype=torch.long, device=device, requires_grad=False)]
        ,
        "tags": tensor_from_dict(dictionnaries["tags"], sentence["tags"], device=device)
    }

    # bert
    if bert_tokenizer is not None:
        words = [pydestruct.nn.bert.BERT_TOKEN_MAPPING.get(word, word) for word in sentence["words"]]
        ret["bert"] = bert_tokenizer(words, boundaries=True)

    if copy_constituents:
        ret["constituents"] = sentence["constituents"] # used for eval, so just put it there if asked

    if constituent_input:
        ret.update(build_constituent_inputs(sentence, dictionnaries, device=device))

    if "ext_words" in dictionnaries:
        ret["ext_words"] = tensor_from_dict(dictionnaries["ext_words"], [t for t in sentence["words"]], device=device)

    return ret
