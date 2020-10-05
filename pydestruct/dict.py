import torch
import pydestruct.input

class Dict:
    def __init__(self, words, unk=None, boundaries=False, pad=False, lower=False):
        self._boundaries = boundaries
        self._unk = unk
        self._lower = lower
        self._word_to_id = dict()
        self._id_to_word = list()

        if pad:
            if "**pad**" in words:
                raise RuntimeError("Pad is already in dict")
            self.pad_index = self._add_word("**pad**")

        if boundaries:
            if "**bos**" in words or "**eos**" in words:
                raise RuntimeError("Boundaries ids are already in dict")
            self._bos = self._add_word("**bos**")
            self._eos = self._add_word("**eos**")

        if unk in words:
            raise RuntimeError("UNK word exists in vocabulary")

        if unk is not None:
            self.unk_index = self._add_word(unk)

        if lower:
            words = set(w.lower() for w in words)
        for word in words:
            self._add_word(word)

    # for internal use only!
    def _add_word(self, word):
        if self._lower:
            word = word.lower()
        id = len(self._id_to_word)
        self._word_to_id[word] = id
        self._id_to_word.append(word)
        return id

    def contains(self, word):
        if self._lower:
            word = word.lower()
        return word in self._word_to_id

    def word_to_id(self, word):
        if self._lower:
            word = word.lower()
        if self._unk is not None:
            return self._word_to_id.get(word, self.unk_index)
        else:
            return self._word_to_id[word]

    def id_to_word(self, id):
        return self._id_to_word[id]

    def __len__(self):
        return len(self._word_to_id)

    def has_unk(self):
        return self._unk is not None

    def has_boundaries(self):
        return self._boundaries

    def bos_id(self):
        return self._bos

    def eos_id(self):
        return self._eos


def build_dictionnaries(data, boundaries=False, char_boundaries=False, postag="tags"):
    dict_labels = set()
    dict_disc_labels = set()
    dict_tags = set()
    dict_words = set()
    dict_chars = set()

    for sentence in data:
        dict_words.update(sentence["words"])
        if postag in sentence:
            dict_tags.update(sentence[postag])
        for word in sentence["words"]:
            dict_chars.update(word)
        if "constituents" in sentence:
            for cst in sentence["constituents"]:
                if len(cst) == 3 or cst[2] < 0:
                    dict_labels.add(cst[0])
                elif len(cst) == 5:
                    dict_disc_labels.add(cst[0])
                else:
                    RuntimeError("Weird constituent: ", cst)

    dict_chars = Dict(dict_chars, boundaries=char_boundaries)
    dict_words = Dict(dict_words, unk="#UNK#", boundaries=boundaries, pad=True)
    dict_tags = Dict(dict_tags, boundaries=boundaries, pad=True)
    dict_labels = Dict(dict_labels)
    dict_disc_labels = Dict(dict_disc_labels)

    return {"chars": dict_chars, "words": dict_words, "labels": dict_labels, "disc_labels": dict_disc_labels, "tags": dict_tags}

