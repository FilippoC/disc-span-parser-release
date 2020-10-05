import torch
import torch.nn as nn
import torch.nn.functional as F

# THIS IS NOT MAINTAINED ANYMORE!

# Most of the code of this class comes from here: https://github.com/nikitakit/self-attentive-parser/blob/acl2018/src/parse_nk.py
# but packed as a nn and without batch support
class FencepostModule(nn.Module):
    # The disentangle model is useful when relying on the output of a transformer.
    # by default, this nn assumes that the output is shaped as a output of a pytorch BiLSTM
    def __init__(self, input_dim, repr_dim, n_labels, disentangle=False, label_bias=True, span_bias=False, activation="tanh"):
        super(FencepostModule, self).__init__()

        self.disentangle = disentangle
        self.activation_name = activation
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            raise RuntimeError("Unknown activation function: %s" % self.activation_name)
        self.label_bias = label_bias
        self.span_bias = span_bias

        self.label_output_mlp = nn.Linear(input_dim, repr_dim, bias=True)
        self.label_output_projection = torch.nn.Parameter(data=torch.Tensor(repr_dim, n_labels))
        # TODO: use a batched MLP as in the dependency nn?
        #       if we do that we will probably need to use custom xavier initialization,
        #       the one of pytorch seems strange for tensors with more the 2 dimensions
        if self.span_bias:
            self.span_output_mlp = nn.Linear(input_dim, repr_dim, bias=True)
            self.span_output_projection = torch.nn.Parameter(data=torch.Tensor(repr_dim, 1))

        if self.label_bias:
            self.output_bias = torch.nn.Parameter(data=torch.Tensor(1, n_labels))

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            if self.activation_name == "tanh":
                torch.nn.init.xavier_uniform_(self.label_output_mlp.weight)
            else:
                torch.nn.init.kaiming_normal_(self.label_output_mlp.weight, nonlinearity=self.activation_name)
            torch.nn.init.zeros_(self.label_output_mlp.bias)
            torch.nn.init.xavier_uniform_(self.label_output_projection)
            if self.span_bias:
                if self.activation_name == "tanh":
                    torch.nn.init.xavier_uniform_(self.span_output_mlp.weight)
                else:
                    torch.nn.init.kaiming_uniform_(self.span_output_mlp.weight, nonlinearity=self.activation_name)
                torch.nn.init.zeros_(self.span_output_mlp.bias)
                torch.nn.init.xavier_uniform_(self.span_output_projection)

            if self.label_bias:
                self.output_bias.fill_(0.0)

    """
        Input dimension is: (n_words + 2,  input_repr)
    """
    def forward(self, input):
        n_words = input.size()[0]
        input_repr = input.size()[1]

        if self.disentangle:
            # when the input comes from a transformer, it may be useful to
            # "disantagle" the representation
            input = torch.cat([
                input[:, 0::2],
                input[:, 1::2],
            ], 1)

        # the context sennsitive embeddings of the span from word i to word j (both included) is:
        # forward[j] - forward[i-1] ; backward[i] - backward[j+1]

        # the BiLSTM output dim is:
        # (seq_len, batch, num_directions * hidden_size):
        # with:
        # - seq_len = n_words + 2 (i.e. including boundary tokens)
        # - batch = 1
        # - num_directions = 2
        #
        # which as dim redim as:
        # (n_words + 2,  hidden_size * 2)
        #
        # Therefore
        # - annotations[:-1, :hidden_size]: all forward hidden dim except for the last token (i.e. eos token)
        # - annotations[1:, hidden_size:]: all backward hidden dim except for the first position (i.e. bos token)
        #
        # fencepost_annotations is of dim:
        # (n_words + 1, hidden_size * 2)
        # where each line i contains [ forward[i] ; -backward[i+1] ]
        # TODO: the minus sign is probably useless, maybe we should remove it for the transformer model?
        fencepost_annotations = torch.cat([
            input[:-1, :input_repr // 2],
            -input[1:, input_repr // 2:],
        ], 1)

        # Remember that we want to compute:
        # forward[j] - forward[i-1] ; backward[i] - backward[j+1]
        # = forward[j] - forward[i-1] ; (-backward[j+1]) - (-backward[i])
        # with variable replacement i' = i-1
        # = forward[j] - forward[i'] ; (-backward[j+1]) - (-backward[i'+1])
        # and we have in fencepost_annotations:
        # [ forward[i] ; -backward[i+1] ]
        # so we can do a substraction with broadcasting

        # span_features dim: (n_words + 1, n_words + 1, hidden_size)
        # where span_features[x, y] = [ forward[x] - forward[y] ; -backward[x+1] - (-backward[y+1]) ]
        #
        # so in order to get the features for the span from word i to j, that is:
        # forward[j] - forward[i-1] ; backward[i] - backward[j+1]
        # we have to get span_features[x, y]:
        # x = j
        # y = i - 1
        # note that i=0 correspond to BOS tag, so it is normal to have y = i-1,
        # there is not representation computed for a span that includes the BOS token,
        # hence you should never set y=-1
        span_features = (
            # (1, n_words + 1, hidden_size)
                torch.unsqueeze(fencepost_annotations, 0)
                -
                # (n_words + 1, 1, hidden_size)
                torch.unsqueeze(fencepost_annotations, 1)
        )

        n_words = span_features.size()[0]
        span_features = span_features.view(n_words * n_words, -1)
        output = self.activation(self.label_output_mlp(span_features)) @ self.label_output_projection
        if self.span_bias:
            output = output + self.activation(self.span_output_mlp(span_features)) @ self.span_output_projection
        if self.label_bias:
            output = output + self.output_bias
        output = output.view(n_words, n_words, -1)

        return output

