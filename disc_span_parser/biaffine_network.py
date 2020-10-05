import torch
import torch.nn as nn
from pydestruct.input import tensor_from_dict
import numpy as np

from pydestruct.network import FeatureExtractionModule
from pydestruct.nn.bilstm import BiLSTM
from pydestruct.nn.dropout import SharedDropout
from pydestruct.nn.dependency import BatchedBiaffine
from pydestruct.dict import Dict
import pydestruct.nn.transformer_klein as transformer_klein

class BiaffineParserNetwork(nn.Module):
    def __init__(self, args, n_cont_labels, n_disc_labels, n_tags=-1, default_lstm_init=False, old=False,**arg_dict):
        super(BiaffineParserNetwork, self).__init__()

        if n_cont_labels == 0 or n_disc_labels == 0:
            raise RuntimeError("Cannot instantiate if number of labels=0")

        self.feature_extractor = FeatureExtractionModule(args, n_tags=n_tags, default_lstm_init=default_lstm_init, **arg_dict)
        w_input_dim = self.feature_extractor.output_dim

        self.bilstms = nn.ModuleList(
            BiLSTM(w_input_dim if i == 0 else args.lstm_dim * 2, args.lstm_dim, num_layers=args.lstm_layers, dropout=args.lstm_dropout)
            for i in range(args.lstm_stacks)
        )
        self.bilstm_dropout = SharedDropout(p=args.lstm_dropout)

        if args.tagger:
            if n_tags <= 0:
                raise RuntimeError("Invalid number of tags")
            self.tagger = nn.Linear(args.lstm_dim * 2, n_tags if old else n_tags-2, bias=True)
            if not (args.tagger_stack >= 1 and args.tagger_stack <= args.lstm_stacks):
                raise RuntimeError("Invalid stack index")
            self.tagger_stack = args.tagger_stack
        else:
            self.tagger = None

        self.label_weights = nn.ModuleDict({
            "cont": BatchedBiaffine(
                        input_dim=args.lstm_dim * 2,
                        proj_dim=args.label_proj_dim,
                        n_labels=n_cont_labels,
                        activation="leaky_relu",
                        dropout=args.mlp_dropout,
                        output_bias=False),
            "disc": BatchedBiaffine(
                        input_dim=args.lstm_dim * 2,
                        proj_dim=args.label_proj_dim,
                        n_labels=n_disc_labels,
                        activation="leaky_relu",
                        dropout=args.mlp_dropout,
                        output_bias=False),
            "gap": BatchedBiaffine(
                        input_dim=args.lstm_dim * 2,
                        proj_dim=args.label_proj_dim,
                        n_labels=n_disc_labels,
                        activation="leaky_relu",
                        dropout=args.mlp_dropout,
                        output_bias=False),
        })
        self.span_weights = nn.ModuleDict({
            "cont": BatchedBiaffine(
                        input_dim=args.lstm_dim * 2,
                        proj_dim=args.span_proj_dim,
                        n_labels=1,
                        activation="leaky_relu",
                        dropout=args.mlp_dropout,
                        output_bias=False),
            "disc": BatchedBiaffine(
                        input_dim=args.lstm_dim * 2,
                        proj_dim=args.span_proj_dim,
                        n_labels=1,
                        activation="leaky_relu",
                        dropout=args.mlp_dropout,
                        output_bias=False),
            "gap": BatchedBiaffine(
                        input_dim=args.lstm_dim * 2,
                        proj_dim=args.span_proj_dim,
                        n_labels=1,
                        activation="leaky_relu",
                        dropout=args.mlp_dropout,
                        output_bias=False),
        })

    def forward(self, input, batched=False):
        features, lengths = self.feature_extractor(input)

        tags = None
        for stack in range(len(self.bilstms)):
            features = torch.nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
            features, _ = self.bilstms[stack](features)
            features, _ = torch.nn.utils.rnn.pad_packed_sequence(features, batch_first=True)
            features = self.bilstm_dropout(features)

            if self.tagger is not None and self.tagger_stack == stack + 1:
                tags = self.tagger(features)

        span_weights = {k: v(features) for k, v in self.span_weights.items()}
        label_weights = {k: v(features) for k, v in self.label_weights.items()}

        # need to break each bach here
        if batched:
            return span_weights, label_weights, tags
        else:
            raise NotImplementedError()

    @staticmethod
    def add_cmd_options(cmd):
        FeatureExtractionModule.add_cmd_options(cmd)

        cmd.add_argument('--biaffine', action="store_true", help="Use biaffine model")
        cmd.add_argument('--proj-dim', type=int, default=200, help="Dimension of the output projection")
        cmd.add_argument('--label-proj-dim', type=int, default=200)
        cmd.add_argument('--span-proj-dim', type=int, default=200)
        cmd.add_argument('--mlp-dropout', type=float, default=0., help="MLP dropout")

        cmd.add_argument('--tagger', action="store_true")
        cmd.add_argument('--tagger-stack', type=int, default=1)

        cmd.add_argument('--lstm-dim', type=int, default=200, help="Dimension of the sentence-level BiLSTM")
        cmd.add_argument('--lstm-layers', type=int, default=1, help="Number of layers of the sentence-level BiLSTM")
        cmd.add_argument('--lstm-stacks', type=int, default=1, help="Number of layers of the sentence-level BiLSTM")
        cmd.add_argument('--lstm-dropout', type=float, default=0., help="BiLSTM dropout")
