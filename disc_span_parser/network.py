import torch.nn as nn
import torch

from pydestruct.network import FeatureExtractionModule, WeightingModule, LSTMWeightingModule
from pydestruct.nn.dependency import DependencyModule
from pydestruct.nn.tagger import TaggerModule


class Network(nn.Module):
    def __init__(self, args, n_labels, n_disc_labels, predict_tags=False, **arg_dict):
        super(Network, self).__init__()

        if args.tagger2 and not predict_tags:
            raise RuntimeError("You asked for the parallel tagger but not for tags!")
        if args.pos_embs and predict_tags:
            raise RuntimeError("You are asking for tag embs AND a tagger! you can use --pos-embs2 with --tagger2 to use build-in tagger")
        if args.pos_embs2 and not args.tagger2:
            raise RuntimeError("Cannot use build in tags without build in tagger!")

        self.feature_extractor = FeatureExtractionModule(args, **arg_dict)
        w_input_dim = self.feature_extractor.output_dim

        self.pos_embs = None
        if args.tagger2:
            outputs = {}

            tagger_args = lambda: None
            tagger_args.lstm_dim = args.tagger_lstm_dim
            tagger_args.lstm_layers = args.tagger_lstm_layers
            tagger_args.lstm_stacks = args.tagger_lstm_stacks
            tagger_args.dropout_lstm_input = args.dropout_lstm_input
            tagger_args.dropout_lstm_output = args.dropout_lstm_output
            tagger_args.dropout_lstm_stack = args.dropout_lstm_stack

            self.tagger_weightning = LSTMWeightingModule(self.feature_extractor.output_dim, tagger_args, outputs=outputs)
            self.tagger = TaggerModule(tagger_args.lstm_dim * 2, arg_dict["n_tags"])

            self.pos_embs = nn.Embedding(arg_dict["n_tags"] + 1, args.pos_embs_dim, padding_idx=arg_dict["n_tags"])
            if args.pos_embs2:
                w_input_dim += args.pos_embs_dim
                self.use_buildin_tags = True
            else:
                self.use_buildin_tags = False
        else:
            self.tagger_weightning = None

        outputs = {
            k: {
                "class": DependencyModule,
                "options": {
                    "model": "biaffine" if args.biaffine else "standard",
                    "proj_dim": args.proj_dim,
                    "n_labels": n_labels if k == "cont" else n_disc_labels,
                    "fake_root": False,
                    "span_bias": args.span_bias,
                    "span_bias_proj": args.span_bias_proj,
                    "activation": args.activation,
                    "dropout": args.proj_dropout
                },
                "layer": -1
            }
            for k in ["cont", "disc", "gap"]
        }

        if predict_tags and not args.tagger2:
            outputs["tags"] = {
                "class": TaggerModule,
                "options": {
                    "n_labels": arg_dict["n_tags"],
                },
                "layer": args.tagger_stack
            }
        self.weightning = WeightingModule(w_input_dim, args, outputs=outputs)
        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            if self.pos_embs is not None:
                nn.init.uniform_(self.pos_embs.weight, -0.01, 0.01)
                if self.pos_embs.padding_idx is not None:
                    self.pos_embs.weight[self.pos_embs.padding_idx].fill_(0)

    def forward(self, input):
        batch_size = len(input)
        feature, length = self.feature_extractor(input)

        if self.tagger_weightning is not None:
            _, tagger_features = self.tagger_weightning(feature, length)
            tagger_features, _ = nn.utils.rnn.pad_packed_sequence(tagger_features, batch_first=True)
            tag_weights = self.tagger(tagger_features)

            if self.use_buildin_tags:
                if self.training:
                    padded_inputs = torch.nn.utils.rnn.pad_sequence(
                        [t["tags"].to(self.pos_embs.weight.device) for t in input],
                        batch_first=True,
                        padding_value=self.pos_embs.padding_idx
                    )
                    feature = torch.cat([feature, self.pos_embs(padded_inputs)], 2)
                else:
                    _, tags = torch.max(tag_weights, dim=2)
                    for b, l in enumerate(length):
                        tags[b, l:] = 0.
                    tag_embs = self.pos_embs(tags)
                    feature = torch.cat([feature, tag_embs], 2)

            ret = self.weightning(feature, length)[0]
            ret["tags"] = [tag_weights[b][:length[b]] for b in range(batch_size)]
            return ret
        else:
            return self.weightning(feature, length)[0]

    @staticmethod
    def add_cmd_options(cmd):
        FeatureExtractionModule.add_cmd_options(cmd)
        WeightingModule.add_cmd_options(cmd)

        cmd.add_argument('--proj-dim', type=int, default=200, help="Dimension of the output projection")
        cmd.add_argument('--proj-dropout', type=float, default=0.)
        cmd.add_argument('--span-bias', action="store_true", help="add a span bias to constituent weights")
        cmd.add_argument('--span-bias-proj', action="store_true", help="use a different projection for the span bias")
        cmd.add_argument('--activation', type=str, default="tanh", help="activation to use in weightning modules: tanh, relu, leaky_relu")
        cmd.add_argument('--tagger-stack', type=int, default="1", help="After which LSTM stack should we put the tagger?")
        cmd.add_argument('--biaffine', action="store_true", help="Use biaffine model")

        cmd.add_argument('--tagger2', action="store_true")
        cmd.add_argument('--pos-embs2', action="store_true")
        cmd.add_argument('--tagger-lstm-dim', type=int, default=200, help="Dimension of the sentence-level BiLSTM")
        cmd.add_argument('--tagger-lstm-layers', type=int, default=1, help="Number of layers of the sentence-level BiLSTM")
        cmd.add_argument('--tagger-lstm-stacks', type=int, default=1, help="Number of stacks of the sentence-level BiLSTM")
