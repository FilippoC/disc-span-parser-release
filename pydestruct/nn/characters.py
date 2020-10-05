import torch
import torch.nn as nn
import pydestruct.nn.dropout

class CharacterEncoder(nn.Module):
    def __init__(self, emb_dim, voc_size, lstm_dim, input_dropout=0., default_lstm_init=False):
        super(CharacterEncoder, self).__init__()

        self.input_dropout = pydestruct.nn.dropout.IndependentDropout(input_dropout)

        self.emb_dim = emb_dim
        self.lstm_dim = lstm_dim
        self.voc_size = voc_size

        self.embeddings = nn.Embedding(voc_size+1, emb_dim, padding_idx=voc_size)
        self.lstm = nn.LSTM(emb_dim, lstm_dim, num_layers=1, bidirectional=True, batch_first=True)

        #self.initialize_parameters(default_lstm_init)

    def initialize_parameters(self, default_lstm_init=False):
        with torch.no_grad():
            # embeddings
            #nn.init.xavier_uniform_(self.embeddings.weight)
            nn.init.uniform_(self.embeddings.weight, -0.01, 0.01)
            if self.embeddings.padding_idx is not None:
                self.embeddings.weight[self.embeddings.padding_idx].fill_(0)

            # LSTM
            # xavier init + set bias to 0 except forget bias to !
            # not that this will actually set the bias to 2 at runtime
            # because the torch implement use two bias, really strange
            if not default_lstm_init:
                for layer in range(self.lstm.num_layers):
                    for name in self.lstm._all_weights[layer]:
                        param = getattr(self.lstm, name)
                        if name.startswith("weight"):
                            nn.init.xavier_uniform_(param)
                        elif name.startswith("bias"):
                            i = param.size(0) // 4
                            param[0:i].fill_(0.)
                            param[i:2 * i].fill_(1.)
                            param[2*i:].fill_(0.)
                        else:
                            raise RuntimeError("Unexpected parameter name in LSTM: %s" % name)

        # init code of Maximin Coavoux
        #self.embeddings.weight.data.uniform_(-embed_init, embed_init)
        #self.embeddings.weight.data[0].fill_(0)
        #for p in self.lstm.parameters():
        #    m = (6 / sum(p.data.shape)) ** 0.5
        #    p.data.uniform_(-m, m)

    def forward(self, input):
        lengths = [len(i) for i in input]
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input,
            batch_first=True,
            padding_value=self.embeddings.padding_idx
        )
        emb_inputs = self.input_dropout(self.embeddings(padded_inputs))[0]

        packed_embs = torch.nn.utils.rnn.pack_padded_sequence(
            emb_inputs,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        _, (endpoints, _) = self.lstm(packed_embs)

        #token_repr = torch.cat([endpoints[0], endpoints[1]], dim=1)
        token_repr = torch.cat(torch.unbind(endpoints), dim=-1)
        return token_repr
