import torch
import torch.nn

def get_scalar_mix():
    from allennlp.modules.scalar_mix import ScalarMix
    return ScalarMix

def get_transformers():
    import transformers
    return transformers

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "#LRB#": "(",
    "#RRB#": ")",
    "#LCB#": "{",
    "#RCB#": "}",
    "#LSB#": "[",
    "#RSB#": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--", # en dash
    "\u2014": "--", # em dash
}

class BertInputBuilder:
    def __init__(self, args):
        if args.bert_model.startswith("camembert"):
            self.tokenizer = get_transformers().CamembertTokenizer.from_pretrained(
                args.bert_tokenizer,
                do_lower_case=args.bert_do_lower_case,
                proxies={"https": args.bert_proxy} if len(args.bert_proxy) > 0 else None,
                cache_dir=args.bert_cache if len(args.bert_cache) > 0 else None
            )
        elif args.bert_model.startswith("flaubert"):
            self.tokenizer = get_transformers().FlaubertTokenizer.from_pretrained(
                args.bert_tokenizer,
                do_lower_case=args.bert_do_lower_case,
                proxies={"https": args.bert_proxy} if len(args.bert_proxy) > 0 else None,
                cache_dir=args.bert_cache if len(args.bert_cache) > 0 else None
            )
        else:
            self.tokenizer = get_transformers().BertTokenizer.from_pretrained(
                args.bert_tokenizer,
                do_lower_case=args.bert_do_lower_case,
                proxies={"https": args.bert_proxy} if len(args.bert_proxy) > 0 else None,
                cache_dir=args.bert_cache if len(args.bert_cache) > 0 else None
            )

        self.split_nt = args.bert_split_nt
        self.start_features = args.bert_start_features
        self.end_feature = args.bert_end_features

    def __call__(self, sentence, boundaries=False, device="cpu"):
        bert_repr = dict()
        bert_repr["n_tokens"] = len(sentence) + (2 if boundaries else 0)

        tokens = []
        word_start_mask = []
        word_end_mask = []

        # contrary to Nikita Kitaev,
        # we don't care about the beggining/end tokens
        # may be useful if we reintroduce a "fencepost" model again
        tokens.append("[CLS]")
        word_start_mask.append(1 if boundaries else 0)
        word_end_mask.append(1 if boundaries else 0)

        # Kitaev and Klein do this trick, so we do it too if requested
        if self.split_nt:
            cleaned_words = []
            for word in sentence:
                if word == "n't" and len(cleaned_words) > 0:
                    cleaned_words[-1] = cleaned_words[-1] + "n"
                    word = "'t"
                cleaned_words.append(word)
        else:
            cleaned_words = sentence

        cleaned_words = [BERT_TOKEN_MAPPING.get(word, word) for word in cleaned_words]

        for word in cleaned_words:
            word_tokens = self.tokenizer.tokenize(word)
            for _ in range(len(word_tokens)):
                word_start_mask.append(0)
                word_end_mask.append(0)
            # as tokens is extended later, overwrite start id with 1
            # (was init to zero just above)
            word_start_mask[len(tokens)] = 1
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)
        # contrary to Nikita Kitaev,
        # we don't care about the beggining/end tokens
        # may be useful if we reintroduce a "fencepost" model again
        tokens.append("[SEP]")
        word_start_mask.append(1 if boundaries else 0)
        word_end_mask.append(1 if boundaries else 0)

        # all ids for the current sentence in the batch
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        bert_repr["input"] = torch.LongTensor(input_ids, device=device)
        if self.start_features:
            bert_repr["start_mask"] = torch.tensor(word_start_mask, dtype=torch.bool, device=device)
        if self.end_feature:
            bert_repr["end_mask"] = torch.tensor(word_end_mask, dtype=torch.bool, device=device)

        return bert_repr


# Code adapted from Kitaev and Klein
class Bert(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        if not (args.bert_start_features or args.bert_end_features):
            raise RuntimeError("No Bert feature selected!")

        if args.bert_mix_layers:
            self.layer_indices = []
            for r in args.bert_mix_layers.split(","):
                r = r.split("-")
                if len(r) == 1:
                    self.layer_indices.append(int(r[0]))
                elif len(r) == 2:
                    self.layer_indices.extend(range(int(r[0]), int(r[1]) + 1))
                else:
                    raise RuntimeError("Invalid format for layer mix")

            self.scalar_mix = get_scalar_mix()(
                len(self.layer_indices),
                do_layer_norm=False,
                initial_scalar_parameters=None,
                trainable=True,
            )
        else:
            self.layer_indices = None

        if args.bert_model.startswith("camembert"):
            self.bert = get_transformers().CamembertModel.from_pretrained(
                args.bert_model,
                proxies={"https": args.bert_proxy} if len(args.bert_proxy) > 0 else None,
                cache_dir=args.bert_cache if len(args.bert_cache) > 0 else None,
                output_hidden_states=self.layer_indices is not None
            )
        elif args.bert_model.startswith("flaubert"):
            self.bert = get_transformers().FlaubertModel.from_pretrained(
                args.bert_model,
                proxies={"https": args.bert_proxy} if len(args.bert_proxy) > 0 else None,
                cache_dir=args.bert_cache if len(args.bert_cache) > 0 else None,
                output_hidden_states=self.layer_indices is not None
            )
        else:
            self.bert = get_transformers().BertModel.from_pretrained(
                args.bert_model,
                proxies={"https": args.bert_proxy} if len(args.bert_proxy) > 0 else None,
                cache_dir=args.bert_cache if len(args.bert_cache) > 0 else None,
                output_hidden_states=self.layer_indices is not None
            )

        self.start_features = args.bert_start_features
        self.end_feature = args.bert_end_features

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--bert-tokenizer', type=str)
        cmd.add_argument('--bert-model', type=str)
        cmd.add_argument('--bert-do-lower-case', action="store_true")
        cmd.add_argument('--bert-start-features', action="store_true")
        cmd.add_argument('--bert-end-features', action="store_true")
        cmd.add_argument('--bert-split-nt', action="store_true")
        cmd.add_argument('--bert-proxy', type=str, default="")
        cmd.add_argument('--bert-cache', type=str, default="")
        cmd.add_argument('--bert-mix-layers', type=str, default="", help="By default the last layer of BERT is used. You can set this to learn a mix of a subset of layers, e.g. 1,4-5,8. The embedding layer is included!")

    def get_output_dim(self):
        return self.bert.pooler.dense.in_features

    def forward(self, sentences, raw_features=False):
        """
        :param sentences: list of list of word (in string format!)
        :return: padded repr (batch size, repr size)
        """

        with torch.no_grad():
            all_input_ids = torch.nn.utils.rnn.pad_sequence([sentence["input"] for sentence in sentences], batch_first=True, padding_value=0)

            all_lengths = torch.tensor([len(sentence["input"]) for sentence in sentences], dtype=torch.long, device=all_input_ids.device)  # length of each sequence
            all_input_mask = torch.arange(all_input_ids.shape[1], device=all_input_ids.device)[None, :] < all_lengths[:, None]

            # by default BERT just returns the last hidden layer
            # (batch_size, sequence_length, hidden_size)
            if self.layer_indices is None:
                features, _ = self.bert(all_input_ids, attention_mask=all_input_mask)
                del _
            else:
                _, __, features = self.bert(all_input_ids, attention_mask=all_input_mask)
                del _
                del __

                features = self.scalar_mix([features[i] for i in self.layer_indices])

        all_features = []
        if "start_mask" in sentences[0]:
            all_word_start_mask = torch.nn.utils.rnn.pad_sequence([sentence["start_mask"] for sentence in sentences], batch_first=True, padding_value=0)
            all_features.append(features.masked_select(all_word_start_mask.unsqueeze(-1)))

        if "end_mask" in sentences[0]:
            all_word_end_mask = torch.nn.utils.rnn.pad_sequence([sentence["end_mask"] for sentence in sentences], batch_first=True, padding_value=0)
            all_features.append(features.masked_select(all_word_end_mask.unsqueeze(-1)))

        if len(all_features) == 1:
            features = all_features[0]
        else:
            features = all_features[0] + all_features[1]

        max_length = max(sentence["n_tokens"] for sentence in sentences)
        features = features.reshape(-1, self.get_output_dim())
        ret_features = torch.zeros((len(sentences), max_length, features.shape[1]), device=features.device)
        start = 0
        for b, size in enumerate(sentence["n_tokens"] for sentence in sentences):
            ret_features[b, 0:size, :] = features[start:start + size, :]
            start += size
        return ret_features
