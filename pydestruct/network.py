import torch.nn as nn
import torch
import itertools
import numpy as np

from pydestruct.nn.bert import Bert
from pydestruct.nn.characters import CharacterEncoder
from pydestruct.nn.transformer import Transformer
from pydestruct.nn.bilstm import BiLSTM
from pydestruct.nn.dependency import DependencyModule
import pydestruct.nn.dropout

def get_elmo():
    import allennlp.modules.elmo as elmo
    return elmo

def update_feature_size(before, new, sum=False):
    if sum:
        if before == 0:
            return new
        else:
            if before != new:
                raise RuntimeError("Invalid feature size")
            else:
                return before
    else:
        return before + new

class FeatureExtractionModule(nn.Module):
    def __init__(self, args, n_chars, n_words, n_tags=0, n_lemmas=0, word_counts=None, ext_word_dict=None, unk_word_index=-1, default_lstm_init=False):
        super(FeatureExtractionModule, self).__init__()

        self.word_dropout = args.word_dropout
        self.word_counts = word_counts
        self.unk_word_index = unk_word_index
        self.sum_features = args.sum_features

        self.output_dropout = pydestruct.nn.dropout.IndependentDropout(args.dropout_features)
        token_feats_dim = 0
        if args.pos_embs:
            if n_tags <= 0:
                raise RuntimeError("Number of tags is not set!")
            token_feats_dim = update_feature_size(token_feats_dim, args.pos_embs_dim, args.sum_features)
            self.pos_embs = nn.Embedding(n_tags + 1, args.pos_embs_dim, padding_idx=n_tags)
        else:
            self.pos_embs = None

        if args.lemma_embs:
            if n_lemmas <= 0:
                raise RuntimeError("Number of lemmas is not set!")
            token_feats_dim = update_feature_size(token_feats_dim, args.lemma_embs_dim, args.sum_features)
            self.lemma_embs = nn.Embedding(n_lemmas + 1, args.lemma_embs_dim, padding_idx=n_lemmas)
        else:
            self.lemma_embs = None

        if args.char_embs:
            token_feats_dim = update_feature_size(token_feats_dim, args.char_lstm_dim * 2, args.sum_features)
            self.char_lstm = CharacterEncoder(args.char_embs_dim, n_chars, args.char_lstm_dim, input_dropout=args.dropout_char_lstm_input, default_lstm_init=default_lstm_init)
        else:
            self.char_lstm = None

        if args.word_embs:
            token_feats_dim = update_feature_size(token_feats_dim, args.word_embs_dim, args.sum_features)
            self.word_embs = nn.Embedding(n_words + 1, args.word_embs_dim, padding_idx=n_words)
        else:
            self.word_embs = None

        if args.pretrained_word_embs:
            if ext_word_dict is None:
                raise RuntimeError("Must pass a word dictionnary when using pretrained embeddings!")
            token_feats_dim = update_feature_size(token_feats_dim, args.pretrained_word_embs_dim, args.sum_features)

            with torch.no_grad():
                embs = torch.zeros(len(ext_word_dict) + 1, args.pretrained_word_embs_dim)
                with open(args.pretrained_word_embs_path, 'r') as f:
                    for line in f:
                        line = line.split()
                        word = line[0]
                        if not ext_word_dict.contains(word):
                            raise RuntimeError("Error in extern dict: %s" % word)
                        vector = list(map(float, line[1:]))
                        embs[ext_word_dict.word_to_id(word)] = torch.tensor(vector)
                embs[ext_word_dict.word_to_id(ext_word_dict._unk)] = 0.
                embs /= torch.std(embs)
            self.pretrained_word_embs = nn.Embedding.from_pretrained(embs, padding_idx=n_words)

            if args.pretrained_word_embs_finetune:
                self.pretrained_word_embs2 = nn.Embedding(n_words + 1, args.word_embs_dim, padding_idx=n_words)
                nn.init.zeros_(self.pretrained_word_embs2.weight)
            else:
                self.pretrained_word_embs2 = None
        else:
            self.pretrained_word_embs = None

        if args.elmo:
            import allennlp.modules.elmo
            self.elmo = get_elmo().Elmo(
                options_file=args.elmo_options,
                weight_file=args.elmo_weights,
                num_output_representations=1,
                requires_grad=False,
                do_layer_norm=False,
                keep_sentence_boundaries=False,
                dropout=0.
            )
            token_feats_dim = update_feature_size(token_feats_dim, self.elmo.get_output_dim(), args.sum_features)

        else:
            self.elmo = None

        if args.bert:
            self.bert = Bert(args)
            # as of now, we always freeze bert
            token_feats_dim = update_feature_size(token_feats_dim, self.bert.get_output_dim(), args.sum_features)
        else:
            self.bert = None

        if token_feats_dim == 0:
            raise RuntimeError("No input features where set!")

        self.output_dim = token_feats_dim

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            if self.word_embs is not None:
                nn.init.uniform_(self.word_embs.weight, -0.01, 0.01)
                if self.word_embs.padding_idx is not None:
                    self.word_embs.weight[self.word_embs.padding_idx].fill_(0)
            if self.pos_embs is not None:
                nn.init.uniform_(self.pos_embs.weight, -0.01, 0.01)
                if self.pos_embs.padding_idx is not None:
                    self.pos_embs.weight[self.pos_embs.padding_idx].fill_(0)
            if self.lemma_embs is not None:
                nn.init.uniform_(self.lemma_embs.weight, -0.01, 0.01)
                if self.lemma_embs.padding_idx is not None:
                    self.lemma_embs.weight[self.lemma_embs.padding_idx].fill_(0)

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--sum-features', action="store_true", help="Sum features")

        cmd.add_argument('--elmo', action="store_true", help="Use ELMO")
        cmd.add_argument('--elmo-options', type=str)
        cmd.add_argument('--elmo-weights', type=str)

        cmd.add_argument('--bert', action="store_true", help="Use Bert")
        Bert.add_cmd_options(cmd)

        cmd.add_argument('--pos-embs', action="store_true", help="Use POS embeddings")
        cmd.add_argument('--pos-embs-dim', type=int, default=50, help="Dimension of the POS embs")

        cmd.add_argument('--char-embs', action="store_true", help="Use a character BiLSTM")
        cmd.add_argument('--char-embs-dim', type=int, default=30, help="Dimension of the character embeddings")
        cmd.add_argument('--char-lstm-dim', type=int, default=100, help="Hidden dimension of the character BiLSTM")

        cmd.add_argument('--word-embs', action="store_true", help="Use word embeddings (randomly initialized)")
        cmd.add_argument('--word-embs-dim', type=int, default=100, help="Dimension of the word embeddings")

        cmd.add_argument('--lemma-embs', action="store_true", help="Use word embeddings (randomly initialized)")
        cmd.add_argument('--lemma-embs-dim', type=int, default=100, help="Dimension of the word embeddings")

        cmd.add_argument('--pretrained-word-embs', action="store_true", help="Use word embeddings (pretrained)")
        cmd.add_argument('--pretrained-word-embs-finetune', action="store_true", help="Use word embeddings (pretrained)")
        cmd.add_argument('--pretrained-word-embs-path', type=str, help="Use word embeddings (pretrained)")
        cmd.add_argument('--pretrained-word-embs-dim', type=int, default=100, help="Dimension of the word embeddings")

        # dropout
        cmd.add_argument('--dropout-features', type=float, default=0.5)
        cmd.add_argument('--dropout-char-lstm-input', type=float, default=0.0, help="Dropout rate to apply at the input of the character BiLSTM")
        cmd.add_argument('--word-dropout', action="store_true", help="Word dropout randomly replace words with the unk token during training with probability 1/(1+count)")

    # flat=False: ret will be of dim (n_batch, max sentence size, feature size)
    # flat=True: ret will be (sum of all sentence sizes, feature size),
    #            ie. all repr of all sentences are concatenated, this is useful for Kitaev transformer
    def forward(self, inputs, flat=False):
        # 1. build token repr for the entire batch
        batch_size = len(inputs)
        lengths = [len(t["words"]) for t in inputs]

        repr_list = []
        # create word representations from table
        if self.word_embs is not None:
            batch_words = [t["words"].to(self.word_embs.weight.device) for t in inputs]

            # todo: this is really inefficient
            # is word dropout still useful anyway?
            if self.training and self.word_dropout:
                batch_words = [words.clone() for words in batch_words]
                for b in range(len(batch_words)):
                    for i in range(batch_words[b].size()[0]):
                        if np.random.rand() < 1 / (1 + self.word_counts[batch_words[b][i].item()]):
                            batch_words[b][i] = self.unk_word_index

            if flat:
                batch_words = torch.cat(batch_words)
                repr_list.append(self.word_embs(batch_words))
            else:
                padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    batch_words,
                    batch_first=True,
                    padding_value=self.word_embs.padding_idx
                )
                repr_list.append(self.word_embs(padded_inputs))

        if self.pretrained_word_embs is not None:
            if flat:
                batch_words = [t["ext_words"].to(self.pretrained_word_embs.weight.device) for t in inputs]
                batch_words = torch.cat(batch_words)
                embs = self.pretrained_word_embs(batch_words)
                if self.pretrained_word_embs2 is not None:
                    batch_words = [t["words"].to(self.pretrained_word_embs2.weight.device) for t in inputs]
                    batch_words = torch.cat(batch_words)
                    embs = embs + self.pretrained_word_embs2(batch_words)
            else:
                batch_words = [t["ext_words"].to(self.pretrained_word_embs.weight.device) for t in inputs]
                padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    batch_words,
                    batch_first=True,
                    padding_value=self.pretrained_word_embs.padding_idx
                )
                embs = self.pretrained_word_embs(padded_inputs)
                if self.pretrained_word_embs2 is not None:
                    batch_words = [t["words"].to(self.pretrained_word_embs2.weight.device) for t in inputs]
                    padded_inputs = torch.nn.utils.rnn.pad_sequence(
                        batch_words,
                        batch_first=True,
                        padding_value=self.pretrained_word_embs2.padding_idx
                    )
                    embs = embs + self.pretrained_word_embs2(padded_inputs)
            repr_list.append(embs)

        if self.pos_embs is not None:
            if flat:
                batch_input = torch.cat([t["tags"].to(self.pos_embs.weight.device) for t in inputs])
                repr_list.append(self.pos_embs(batch_input))
            else:
                padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    [t["tags"].to(self.pos_embs.weight.device) for t in inputs],
                    batch_first=True,
                    padding_value=self.pos_embs.padding_idx
                )
                repr_list.append(self.pos_embs(padded_inputs))

        if self.lemma_embs is not None:
            if flat:
                batch_input = torch.cat([t["lemma"].to(self.lemma_embs.weight.device) for t in inputs])
                repr_list.append(self.lemma_embs(batch_input))
            else:
                padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    [t["lemmas"].to(self.lemma_embs.weight.device) for t in inputs],
                    batch_first=True,
                    padding_value=self.lemma_embs.padding_idx
                )
                repr_list.append(self.lemma_embs(padded_inputs))

        if self.elmo is not None:
            if flat:
                raise RuntimeError("Not implemented yet")
            elmo_inputs = [t["elmo"] for t in inputs]
            elmo_inputs = get_elmo().batch_to_ids(elmo_inputs).to(self.elmo.scalar_mix_0.gamma.device)
            repr_list.append(self.elmo(elmo_inputs)['elmo_representations'][0])

        if self.bert is not None:
            if flat:
                raise RuntimeError("Not implemented yet")
            bert_inputs = [{k: (v.to(self.bert.bert.embeddings.word_embeddings.weight.device) if k != "n_tokens" else v) for k, v in t["bert"].items()} for t in inputs]
            repr_list.append(self.bert(bert_inputs))

        # create word representations from char lstm
        char_token_repr = None
        if self.char_lstm:
            all_char_inputs = list(itertools.chain.from_iterable([v.to(self.char_lstm.embeddings.weight.device) for v in t["chars"]] for t in inputs))
            char_repr = self.char_lstm(all_char_inputs)
            if flat:
                repr_list.append(char_repr)
            else:
                cum_lengths = np.cumsum(lengths)
                char_repr = [char_repr[cum_lengths[i - 1] if i > 0 else 0 : v] for i, v in enumerate(cum_lengths)]
                repr_list.append(torch.nn.utils.rnn.pad_sequence(
                    char_repr,
                    batch_first=True,
                    padding_value=0
                ))

        # apply special dropout
        repr_list = self.output_dropout(*repr_list)

        # combine word representations
        if len(repr_list) == 1:
            token_repr = repr_list[0]
        else:
            if self.sum_features:
                token_repr = sum(repr_list)
            else:
                token_repr = torch.cat(repr_list, 1 if flat else 2)

        return token_repr, lengths


class WeightingModule(nn.Module):
    def __init__(self, token_feats_dim, args, outputs, default_lstm_init=False):
        super(WeightingModule, self).__init__()

        if args.weightning_module == "lstm":
            self.module = LSTMWeightingModule(token_feats_dim, args, outputs, default_lstm_init=default_lstm_init)
        elif args.weightning_module == "transformer":
            self.module = TransformerWeightingModule(token_feats_dim, args, outputs)

    @staticmethod
    def add_cmd_options(cmd):
        LSTMWeightingModule.add_cmd_options(cmd)
        TransformerWeightingModule.add_cmd_options(cmd)
        cmd.add_argument('--weightning-module', type=str, default="lstm")

    def forward(self, inputs, lengths):
        return self.module(inputs, lengths)


class LSTMWeightingModule(nn.Module):
    def __init__(self, token_feats_dim, args, outputs, default_lstm_init=False):
        super(LSTMWeightingModule, self).__init__()

        self.residual = args.lstm_residual
        self.dropout_lstm_output = pydestruct.nn.dropout.SharedDropout(args.dropout_lstm_output)

        if args.lstm_norm:
            self.norm = nn.LayerNorm(token_feats_dim)
        else:
            self.norm = None

        if args.lstm_custom:
            self.custom_lstm = True
            self.lstms = nn.ModuleList([
                BiLSTM(
                    token_feats_dim if i == 0 else args.lstm_dim * 2,
                    args.lstm_dim,
                    num_layers=args.lstm_layers,
                )
                for i in range(args.lstm_stacks)
            ])
        else:
            self.custom_lstm = False
            self.lstms = nn.ModuleList([
                nn.LSTM(
                    token_feats_dim if i == 0 else args.lstm_dim * 2,
                    args.lstm_dim,
                    num_layers=args.lstm_layers,
                    bidirectional=True,
                    batch_first=True
                )
                for i in range(args.lstm_stacks)
            ])

        self.lstms_h = nn.ParameterList([
            torch.nn.Parameter(torch.FloatTensor(args.lstm_layers * 2, 1, args.lstm_dim))
            for _ in range(args.lstm_stacks)
        ])
        self.lstms_c = nn.ParameterList([
            torch.nn.Parameter(torch.FloatTensor(args.lstm_layers * 2, 1, args.lstm_dim))
            for _ in range(args.lstm_stacks)
        ])

        sentence_feats_dim = args.lstm_dim * 2

        modules = [torch.nn.ModuleDict() for _ in range(len(self.lstms) + 1)]
        for name, output_args in outputs.items():
            module_class = output_args["class"]
            module_opts = output_args["options"]
            layer = output_args.get("layer", -1)

            modules[layer][name] = module_class(sentence_feats_dim, **module_opts)
        self.output_modules = torch.nn.ModuleList(modules)

        self.initialize_parameters(default_lstm_init)

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--lstm-dim', type=int, default=200, help="Dimension of the sentence-level BiLSTM")
        cmd.add_argument('--lstm-layers', type=int, default=1, help="Number of layers of the sentence-level BiLSTM")
        cmd.add_argument('--lstm-stacks', type=int, default=2, help="Number of stacks of the sentence-level BiLSTM")
        cmd.add_argument('--lstm-residual', action="store_true")
        cmd.add_argument('--lstm-norm', action="store_true")
        cmd.add_argument('--lstm-custom', action="store_true")
        cmd.add_argument('--dropout-lstm-output', type=float, default=0.3, help="Dropout rate to apply at the output of the sentence-level BiLSTM")

    def initialize_parameters(self, default_lstm_init=False):
        # LSTM
        # xavier init + set bias to 0 except forget bias to !
        # not that this will actually set the bias to 2 at runtime
        # because the torch implement use two bias, really strange
        with torch.no_grad():
            if (not self.custom_lstm) and (not default_lstm_init):
                for lstm in self.lstms:
                    for layer in range(lstm.num_layers):
                        for name in lstm._all_weights[layer]:
                            param = getattr(lstm, name)
                            if name.startswith("weight"):
                                nn.init.xavier_uniform_(param)
                            elif name.startswith("bias"):
                                i = param.size(0) // 4
                                param[0:i].fill_(0.)
                                param[i:2 * i].fill_(1.)
                                param[2 * i:].fill_(0.)
                            else:
                                raise RuntimeError("Unexpected parameter name in LSTM: %s" % name)

            # is there a better way to init this?
            for p in self.lstms_h:
                nn.init.uniform_(p.data, -0.1, 0.1)
            for p in self.lstms_c:
                nn.init.uniform_(p.data, -0.1, 0.1)

    def forward(self, token_repr, lengths):
        # 2. build context sensitive repr and outpus
        batch_size = token_repr.shape[0]

        ret = self.compute_layer(token_repr, lengths, 0)

        for i, lstm in enumerate(self.lstms):
            previous = token_repr

            token_repr = torch.nn.utils.rnn.pack_padded_sequence(token_repr, lengths, batch_first=True, enforce_sorted=False)
            token_repr, _ = lstm(token_repr,
                                 (self.lstms_h[i].repeat(1, batch_size, 1), self.lstms_c[i].repeat(1, batch_size, 1)))
            token_repr, _ = torch.nn.utils.rnn.pad_packed_sequence(token_repr, batch_first=True)

            if self.residual:
                token_repr = token_repr + previous.data
            if self.norm is not None:
                token_repr = self.norm(token_repr)
            if len(self.output_modules[1 + i]) > 0:
                ret.update(self.compute_layer(self.dropout_lstm_output(token_repr), lengths, 1 + i))

        return ret, token_repr

    def compute_layer(self, repr, lengths, index):
        ret = dict()
        modules = self.output_modules[index]
        if len(modules) > 0:
            ret = {name: list() for name in modules.keys()}
            for batch, length in enumerate(lengths):
                t = repr[batch, 0:length]
                for name, module in modules.items():
                    ret[name].append(module(t))
        return ret


class TransformerWeightingModule(nn.Module):
    def __init__(self, token_feats_dim, args, outputs):
        super(TransformerWeightingModule, self).__init__()

        self.dropout_input = nn.Dropout(args.dropout_transformer_input)
        self.dropout_output = nn.Dropout(args.dropout_transformer_output)

        self.transformer = Transformer(args, token_feats_dim)

        self.output_modules = dict()
        for name, output_args in outputs.items():
            module_class = output_args["class"]
            module_opts = output_args["options"]
            layer = output_args.get("layer", -1)
            if layer != -1:
                raise RuntimeError("The only valid layer index for the transformer is -1")

            self.output_modules[name] = module_class(self.transformer.output_dim, **module_opts)

    @staticmethod
    def add_cmd_options(cmd):
        Transformer.add_cmd_options(cmd)
        cmd.add_argument('--dropout-transformer-input', type=float, default=0.3)
        cmd.add_argument('--dropout-transformer-output', type=float, default=0.3)

    def forward(self, token_repr, lengths):
        # 2. build context sensitive repr and outpus

        token_repr = self.dropout_input(token_repr)
        token_repr = self.transformer(token_repr, lengths)
        token_repr = token_repr.transpose(0, 1)

        ret = self.compute_layer(self.dropout_output(token_repr), lengths)

        return ret, token_repr

    def compute_layer(self, repr, lengths):
        ret = dict()
        if len(self.output_modules) > 0:
            ret = {name: list() for name in self.output_modules.keys()}
            for batch, length in enumerate(lengths):
                t = repr[batch, 0:length]
                for name, module in self.output_modules.items():
                    ret[name].append(module(t))
        return ret

