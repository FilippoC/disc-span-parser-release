import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from pydestruct.nn.dropout import SharedDropout, IndependentDropout

import pydestruct.nn.transformer_klein

class LayerNorm(nn.Module):
    def __init__(self, dim, mean=0., std=1., fixed=False, eps=1e-6, ball=False):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.ball = ball

        if fixed:
            self.target_mean = mean
            self.target_std = std
        else:
            self.target_mean = nn.Parameter(torch.empty(dim).fill_(mean))
            self.target_std = nn.Parameter(torch.empty(dim).fill_(std))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(torch.mean((x - mean).pow(2), dim=-1, keepdim=True) + self.eps)
        if self.ball:
            std = std.clamp(1.)
        return self.target_std * (x - mean) / std + self.target_mean

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads, query_dim, values_dim, output_dim, att_proj_bias=False, att_dropout=0.):
        super().__init__()

        self.n_heads = n_heads
        self.query = nn.Linear(input_dim, n_heads * query_dim, bias=att_proj_bias)
        self.key = nn.Linear(input_dim, n_heads * query_dim, bias=att_proj_bias)
        self.value = nn.Linear(input_dim, n_heads * values_dim, bias=att_proj_bias)

        # For attention dropout we use "standard" dropout
        # we should take into account the mask when using att!
        self.att_dropout = nn.Dropout(att_dropout)
        self.temper = query_dim ** 0.5

        self.output_proj = nn.Linear(n_heads * values_dim, output_dim)

    def forward(self, input_features, mask=None, mask_value=-float('inf')):
        batch_size = input_features.shape[0]
        sentence_size = input_features.shape[1]
        # (batch size, sentence size, proj size * n head)
        query = self.query(input_features)
        key = self.key(input_features)

        # (batch size, sentence size, values size * n_head)
        values = self.value(input_features)

        # (batch size, n heads, sentence size, proj size)
        query = query.reshape(batch_size, query.shape[1], self.n_heads, -1).transpose(1, 2)
        key = key.reshape(batch_size, key.shape[1], self.n_heads, -1).transpose(1, 2)
        values = values.reshape(batch_size, values.shape[1], self.n_heads, -1).transpose(1, 2)

        # (batch size, n heads, sentence size, sentence size)
        att_scores = query @ key.transpose(-1, -2)
        att_scores /= self.temper

        if mask is not None:
            att_scores.data.masked_fill_(mask.reshape(att_scores.shape[0], 1, 1, att_scores.shape[-1]), mask_value)

        # compute attention
        att = F.softmax(att_scores, dim=-1)
        att = self.att_dropout(att)

        # aggregate values:
        # the softmax dim is the last dim and the sentence dim is last-1 dim,
        # so this is ok
        # (batch size, n head, sentence size, values size)

        values = att @ values
        # now we must concatenate all heads
        # (batch size, sentence size, values size * n heads)
        values = values.transpose(1, 2).reshape(batch_size, sentence_size, -1)
        values = self.output_proj(values)

        return values


# I can't see the layer norm in the paper (sec. 3.3)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, shared_dropout=True):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

        if shared_dropout:
            # we can use it because input is (batch, n word, n features)
            self.relu_dropout = SharedDropout(dropout)
        else:
            self.relu_dropout = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.relu_dropout(self.relu(self.w_1(x)))
        output = self.w_2(inter)
        return output


class Layer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 query_dim,
                 values_dim,
                 ff_hidden_dim,
                 att_dropout=0.,
                 ff_dropout=0.,
                 residual_att_dropout=0.,
                 residual_ff_dropout=0.,
                 att_proj_bias=False,
                 shared_dropout=True,
                 pre_ln=False,
                 ball_norm=False
         ):
        super(Layer, self).__init__()

        self.self_attn = MultiHeadAttention(
            input_dim=input_dim,
            n_heads=n_heads,
            query_dim=query_dim,
            values_dim=values_dim,
            output_dim=input_dim,  # must be the same size because of the residual connection
            att_dropout=att_dropout,
            att_proj_bias=att_proj_bias
        )
        self.feed_forward = PositionwiseFeedForward(
            input_dim,
            hidden_dim=ff_hidden_dim,
            dropout=ff_dropout,
            shared_dropout=shared_dropout
        )

        self.layer_norm1 = LayerNorm(input_dim, ball=ball_norm)
        self.layer_norm2 = LayerNorm(input_dim, ball=ball_norm)
        self.pre_ln = pre_ln

        if shared_dropout:
            # ok because input is (batch, n word, features)
            self.dropout1 = SharedDropout(residual_att_dropout)
            self.dropout2 = SharedDropout(residual_ff_dropout)
        else:
            self.dropout1 = nn.Dropout(residual_att_dropout)
            self.dropout2 = nn.Dropout(residual_ff_dropout)

    def forward(self, x, mask=None, mask_value=-float('inf')):
        if self.pre_ln:
            x1 = self.layer_norm1(x)
            x1 = self.self_attn(x1, mask=mask, mask_value=mask_value)
            x1 = self.dropout1(x1) + x

            x2 = self.layer_norm2(x1)
            x2 = self.feed_forward(x2)
            x2 = self.dropout2(x2) + x1

        else:
            x1 = self.self_attn(x, mask=mask, mask_value=mask_value)
            x1 = self.dropout1(x1) + x
            x1 = self.layer_norm1(x1)

            x2 = self.feed_forward(x1)
            x2 = self.dropout2(x2) + x1
            x2 = self.layer_norm2(x2)

        return x2


class Transformer(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 n_heads,
                 ff_dim_hidden,
                 query_dim,
                 values_dim,
                 att_dropout=0.,
                 ff_dropout=0.,
                 residual_att_dropout=0.,
                 residual_ff_dropout=0.,
                 norm_input=True,
                 att_proj_bias=False,
                 shared_dropout=True,
                 pre_ln=False,
                 ball_norm=False
                ):
        super().__init__()

        self.layer_norm = LayerNorm(input_dim, ball=ball_norm) if norm_input else None
        self.layers = nn.ModuleList([
            Layer(
                input_dim=input_dim,
                n_heads=n_heads,
                query_dim=query_dim,
                values_dim=values_dim,
                ff_hidden_dim=ff_dim_hidden,
                att_dropout=att_dropout,
                ff_dropout=ff_dropout,
                residual_att_dropout=residual_att_dropout,
                residual_ff_dropout=residual_ff_dropout,
                att_proj_bias=att_proj_bias,
                shared_dropout=shared_dropout,
                pre_ln=pre_ln,
                ball_norm=ball_norm
            )
            for _ in range(num_layers)
        ])

    # mask: but be false at word position and true at masked positions
    def forward(self, emb, lengths=None, mask=None, mask_value=-float('inf')):
        if lengths is not None and mask is None:
            # build mask from lengths
            mask = torch.arange(emb.shape[1], device=emb.device).reshape(1, -1) >= lengths.reshape(-1, 1)

        if self.layer_norm is not None:
            emb = self.layer_norm(emb)

        for i in range(len(self.layers)):
            emb = self.layers[i](emb, mask=mask, mask_value=mask_value)

        return emb


class TransformerNetwork(nn.Module):
    def __init__(self, args, d_input):
        super(TransformerNetwork, self).__init__()

        if args.transformer_position_embs:
            d = args.transformer_dmodel - d_input if args.transformer_position_concat else d_input
            self.position_table = nn.Parameter(torch.FloatTensor(args.transformer_position_max, d))

            if args.transformer_position_concat:
                d_input += d
            self.position_concat = args.transformer_position_concat
            self.position_dropout = IndependentDropout(args.transformer_position_dropout)
        else:
            self.position_table = None

        if d_input != args.transformer_dmodel:
            raise RuntimeError("Input size mismatch: %i - %i" % (d_input, args.transformer_dmodel))

        self.encoder = Transformer(
                 num_layers=args.transformer_n_layers,
                 input_dim=d_input,
                 n_heads=args.transformer_n_heads,
                 ff_dim_hidden=args.transformer_ff_hidden_dim,
                 query_dim=args.transformer_query_dim,
                 values_dim=args.transformer_value_dim,
                 att_dropout=args.transformer_att_dropout,
                 ff_dropout=args.transformer_ff_dropout,
                 residual_att_dropout=args.transformer_residual_att_dropout,
                 residual_ff_dropout=args.transformer_residual_ff_dropout,
                 norm_input=args.transformer_norm_input,
                 att_proj_bias=args.transformer_att_proj_bias,
                 shared_dropout=args.transformer_shared_dropout,
                 pre_ln=args.transformer_pre_ln,
                 ball_norm=args.transformer_ball_norm
                )
        """
        self.encoder = pydestruct.nn.transformer_klein.Encoder(
            d_model=d_input,
            num_layers=args.transformer_n_layers,
            num_heads=args.transformer_n_heads,
            d_kv = args.transformer_query_dim,
            d_ff=args.transformer_ff_hidden_dim,
            d_positional=None,
            num_layers_position_only=0,
            relu_dropout=args.transformer_residual_ff_dropout,
            residual_dropout=args.transformer_residual_att_dropout,
            attention_dropout=args.transformer_att_dropout
        )
        """

        self._reset_parameters()

    def _reset_parameters(self):
        if self.position_table is not None:
            torch.nn.init.uniform_(self.position_table, -0.01, 0.01)

    def forward(self, features, lengths=None):
        if self.position_table is not None:
            # unsqueeze for batch dim
            positions = self.position_dropout(self.position_table[:features.shape[1], :].unsqueeze(0))[0]
            if self.position_concat:
                features = torch.cat([features, positions.expand((features.shape[0], -1, -1))], dim=2)
            else:
                features = features + positions
        output = self.encoder(features, lengths=lengths)

        return output

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--transformer-dmodel', type=int, default=1024)

        cmd.add_argument('--transformer-n-heads', type=int, default=8)
        cmd.add_argument('--transformer-n-layers', type=int, default=8)
        cmd.add_argument('--transformer-ff-hidden-dim', type=int, default=2048)
        cmd.add_argument('--transformer-query-dim', type=int, default=512)
        cmd.add_argument('--transformer-value-dim', type=int, default=512)
        cmd.add_argument('--transformer-att-dropout', type=float, default=0.0)
        cmd.add_argument('--transformer-ff-dropout', type=float, default=0.0)
        cmd.add_argument('--transformer-residual-att-dropout', type=float, default=0.0)
        cmd.add_argument('--transformer-residual-ff-dropout', type=float, default=0.0)
        cmd.add_argument('--transformer-norm-input', type=bool, default=True)
        cmd.add_argument('--transformer-att-proj-bias', type=bool, default=False)
        cmd.add_argument('--transformer-shared-dropout', type=bool, default=True)
        cmd.add_argument('--transformer-pre-ln', action="store_true")

        cmd.add_argument('--transformer-position-embs', action="store_true")
        cmd.add_argument('--transformer-position-concat', action="store_true")
        cmd.add_argument('--transformer-position-max', type=int, default=300)
        cmd.add_argument('--transformer-position-dropout', type=float, default=0.0)
        cmd.add_argument('--transformer-ball-norm', action="store_true")
