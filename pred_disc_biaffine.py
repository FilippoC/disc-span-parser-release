import argparse
import torch.optim
import sys

import pydestruct.data.ptb
import pydestruct.input
import disc_span_parser.network
import pydestruct.eval
import disc_span_parser
import pydestruct.data.export
import cpp_disc_span_parser
from pydestruct.timer import Timer
from disc_span_parser.biaffine_network import BiaffineParserNetwork
from disc_span_parser.biaffine_input import build_torch_input
from pydestruct.batch import batch_iterator_factory

# Read command line
cmd = argparse.ArgumentParser()
cmd.add_argument("--data", type=str, required=True)
cmd.add_argument("--model", type=str, required=True)
cmd.add_argument('--storage-device', type=str, default="cpu")
cmd.add_argument('--device', type=str, default="cpu")
cmd.add_argument('--complexity', type=int, default=6)
cmd.add_argument('--wellnested', action="store_true")
cmd.add_argument("--batch", type=int, default=10, help="Mini-batch size")
cmd.add_argument("--batch-clusters", type=int, default=-1, help="If set, the batch is computed in number of words!")
cmd.add_argument("--pipeline", action="store_true")
cmd.add_argument("--old", action="store_true")
cmd.add_argument('--cubic', action="store_true")
cmd.add_argument('--timing', type=str, default="")
args = cmd.parse_args()


print("Loading network from path: %s" % args.model, file=sys.stderr)
model = torch.load(args.model, map_location=torch.device('cpu'))
dictionnaries = model["dicts"]
network = BiaffineParserNetwork(
    model["args"],
    n_chars=len(dictionnaries["chars"]),
    n_words=len(dictionnaries["words"]),
    n_tags=len(dictionnaries["tags"]),
    n_cont_labels=len(dictionnaries["cont_labels"]),
    n_disc_labels=len(dictionnaries["disc_labels"]),
    old=args.old,
    ext_word_dict=dictionnaries["ext_words"] if "ext_words" in dictionnaries else None
)
network.to(device=args.device)
# strict should be turned to false if we decide to use pre-trained embeddings
network.load_state_dict(model["model_state_dict"], strict=True)
network.eval()


print("Reading dev data located at: %s" % args.data, file=sys.stderr)
data = pydestruct.data.export.read(
    args.data
)


#print("Converting data", file=sys.stderr)
if model["args"].bert:
    bert_tokenizer = pydestruct.nn.bert.BertInputBuilder(model["args"])
else:
    bert_tokenizer = None

torch_data = [
    build_torch_input(sentence, dictionnaries, device=args.storage_device, max_word_len=model["args"].max_word_len, copy_constituents=True, constituent_input=False, bert_tokenizer=bert_tokenizer)
    for sentence in data
]
for i in range(len(torch_data)):
    torch_data[i]["id"] = i

data_iterator = batch_iterator_factory(
    torch_data,
    args.batch,
    n_clusters=args.batch_clusters,
    size_getter=(lambda x: len(x["words"])) if args.batch_clusters > 0 else None,
    shuffle=True
)


print("Loading cpp lib", file=sys.stderr)
max_sentence_size = max(len(sentence["words"]) for sentence in data)
if args.cubic:
    disc_span_parser.load_cpp_lib_cubic(
        max_sentence_size,
        len(dictionnaries["disc_labels"])
    )
else:
    disc_span_parser.load_cpp_lib(
        max_sentence_size,
        argmax_disc=True,
        n_charts=1
)

predicted_trees = []

if args.pipeline:
    raise RuntimeError("Not implemented")

timer_nn = Timer()
timer_parser = Timer()
timer_nn.reset(restart=False)
timer_parser.reset(restart=False)
if len(args.timing) > 0:
    timing_file = open(args.timing, "w")
else:
    timing_file = None
with torch.no_grad():
    for inputs in data_iterator:
        # compute stuff
        timer_nn.start()
        span_weights, label_weights, _ = network(
            inputs,
            batched=True
        )
        timer_nn.stop()

        for b, sentence in enumerate(inputs):
            timer_nn.start()
            n_words = len(inputs[b]["words"]) - 2
            cont_spans = span_weights["cont"][b, 1:n_words + 1, 1:n_words + 1]
            cont_labels = label_weights["cont"][b, 1:n_words + 1, 1:n_words + 1]
            disc_spans = span_weights["disc"][b, 1:n_words + 1, 1:n_words + 1]
            disc_labels = label_weights["disc"][b, 1:n_words + 1, 1:n_words + 1]
            gap_spans = span_weights["gap"][b, 1:n_words + 1, 1:n_words + 1]
            gap_labels = label_weights["gap"][b, 1:n_words + 1, 1:n_words + 1]

            cont_weights = (cont_spans + cont_labels).contiguous().cpu()
            disc_weights = (disc_spans + disc_labels).contiguous().cpu()
            gap_weights = (gap_spans + gap_labels).contiguous().cpu()
            timer_nn.stop()

            # TODO: replace tags with predicted tags
            timer_parser.start()
            timer = Timer()
            timer.reset(restart=True)
            if args.cubic:
                predicted_trees.append(cpp_disc_span_parser.argmax_cubic_as_sentence_ptr(
                    disc_span_parser.ARGMAX_CUBIC_CHART,
                    data[sentence["id"]],
                    dictionnaries["cont_labels"]._id_to_word,
                    dictionnaries["disc_labels"]._id_to_word,
                    cont_weights.data_ptr(),
                    disc_weights.data_ptr(),
                    gap_weights.data_ptr()
                ))
            else:
                predicted_trees.append(cpp_disc_span_parser.argmax_disc_as_sentence_ptr(
                    disc_span_parser.ARGMAX_DISC_CHART,
                    data[sentence["id"]],
                    dictionnaries["cont_labels"]._id_to_word,
                    dictionnaries["disc_labels"]._id_to_word,
                    cont_weights.data_ptr(),
                    disc_weights.data_ptr(),
                    gap_weights.data_ptr(),
                    args.complexity,
                    not args.wellnested
                ))
            timer.stop()
            timer_parser.stop()

            if timing_file is not None:
                timing_file.write("%i\t%.f\n" % (n_words, timer.seconds()))

if timing_file is not None:
    timing_file.close()
print("NN time (seconds): ", timer_nn.seconds(), file=sys.stderr)
print("Parsing time (seconds): ", timer_parser.seconds(), file=sys.stderr)
cpp_disc_span_parser.write_data(predicted_trees)
