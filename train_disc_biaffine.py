from __future__ import print_function
import argparse
import torch.optim
import random
import sys

import torch.nn as nn
import pydestruct.data.export
import disc_span_parser
import pydestruct.timer
from disc_span_parser.biaffine_network import BiaffineParserNetwork
from disc_span_parser.biaffine_input import build_torch_input, build_dictionnaries
import pydestruct.nn.bert
import pydestruct.logger
from pydestruct.optim import MetaOptimizer
from pydestruct.batch import batch_iterator_factory
#from disc_span_parser.biaffine_loss import CorrectedBatchUnstructuredProbLoss, BatchUnstructuredApproximateProbLoss, MarginLoss, BatchUnstructuredCorrectProbLoss
from disc_span_parser.biaffine_loss import BatchUnstructuredApproximateProbLoss
import pydestruct.eval
import disc_span_parser.biaffine_eval

def print_log(msg):
    print(msg, file=sys.stderr)
    sys.stderr.flush()

# Read command line
cmd = argparse.ArgumentParser()
cmd.add_argument("--train", type=str, required=True, help="Path to training data")
cmd.add_argument("--dev", type=str, required=True, help="Path to dev data")
cmd.add_argument("--test", type=str, default="", required=False, help="Path to test data")
cmd.add_argument("--model", type=str, required=True, help="Path where to store the model")
cmd.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
cmd.add_argument("--min-word-freq", type=int, default=1)
cmd.add_argument("--max-word-len", type=int, default=-1)
cmd.add_argument("--max-train-len", type=int, default=-1)
cmd.add_argument("--batch", type=int, default=10, help="Mini-batch size")
cmd.add_argument("--batch-clusters", type=int, default=-1, help="If set, the batch is computed in number of words!")
cmd.add_argument('--storage-device', type=str, default="cpu", help="Device where to store the data. It is useful to keep it on CPU when the dataset is large, even computation is done on GPU")
cmd.add_argument('--device', type=str, default="cpu", help="Device to use for computation")
cmd.add_argument('--char-lstm-boundaries', action="store_true", help="Add sentence boundaries at the input of the character BiLSTM")
cmd.add_argument('--tensorboard', type=str, default="", help="Tensorboard path")
cmd.add_argument('--default-lstm-init', action="store_true", help="")
cmd.add_argument('--pipeline', action="store_true", help="")
cmd.add_argument("--mean-loss", action="store_true")
#cmd.add_argument("--nll-loss", action="store_true")
#cmd.add_argument("--margin-loss", action="store_true")
cmd.add_argument("--complexity", type=int, default=5)
cmd.add_argument("--ill-nested", action="store_true")
BiaffineParserNetwork.add_cmd_options(cmd)
MetaOptimizer.add_cmd_options(cmd)
args = cmd.parse_args()

#if args.nll_loss and args.margin_loss:
#    raise RuntimeError("You can only choose one loss")

if len(args.tensorboard) > 0:
    print_log("Tensorboard logging: %s" % args.tensorboard)
    pydestruct.logger.open(args.tensorboard)

print_log("Reading train data located at: %s" % args.train)
train_data = pydestruct.data.export.read(
    args.train
)

print_log("Reading dev data located at: %s" % args.dev)
dev_data = pydestruct.data.export.read(
    args.dev
)

if len(args.test) > 0:
    print_log("Reading test data located at: %s" % args.test)
    test_data = pydestruct.data.export.read(
        args.test
    )
else:
    test_data = []

if args.max_train_len > 0:
    before = len(train_data)
    train_data = [sentence for sentence in train_data if len(sentence["words"]) <= args.max_train_len]
    print_log("N removed sentences in train data: %i" % (before - len(train_data)))

# TODO: do this somewhere else
#for sentence in itertools.chain(train_data, dev_data):
#    words = sentence["tokens"]["form"].copy()
#    for i in range(len(words)):
#        word = words[i]
#        words[i] = pydestruct.nn.bert.BERT_TOKEN_MAPPING.get(word, word)
#    sentence["words"] = words

# we need to remove unreachable constituents first

print_log("Building dictionnaries from train data")
dictionnaries = build_dictionnaries(
    train_data,
    char_boundaries=args.char_lstm_boundaries,
    min_word_freq=args.min_word_freq,
    external=(args.pretrained_word_embs_path if args.pretrained_word_embs else None)
)


print_log("\tn. words: %i" % len(dictionnaries["words"]))
print_log("\tn. chars: %i" % len(dictionnaries["chars"]))
print_log("\tn. cont labels: %i" % len(dictionnaries["cont_labels"]))
print_log("\tn. disc labels: %i" % len(dictionnaries["disc_labels"]))


# we build the torch object rightaway
print_log("Converting data")
if args.bert:
    bert_tokenizer = pydestruct.nn.bert.BertInputBuilder(args)
else:
    bert_tokenizer = None

torch_train = [
    build_torch_input(sentence, dictionnaries, device=args.storage_device, max_word_len=args.max_word_len, bert_tokenizer=bert_tokenizer)
    for sentence in train_data
]

torch_dev = [
    build_torch_input(sentence, dictionnaries, device=args.storage_device, max_word_len=args.max_word_len, copy_constituents=True, bert_tokenizer=bert_tokenizer, constituent_input=False)
    for sentence in dev_data
]
torch_test = [
    build_torch_input(sentence, dictionnaries, device=args.storage_device, max_word_len=args.max_word_len, copy_constituents=True, bert_tokenizer=bert_tokenizer, constituent_input=False)
    for sentence in test_data
]

train_data_iterator = batch_iterator_factory(
    torch_train,
    args.batch,
    n_clusters=args.batch_clusters,
    size_getter=(lambda x: len(x["words"])) if args.batch_clusters > 0 else None,
    shuffle=True
)

dev_data_iterator = batch_iterator_factory(
    torch_dev,
    args.batch,
    n_clusters=args.batch_clusters,
    size_getter=(lambda x: len(x["words"])) if args.batch_clusters > 0 else None,
    shuffle=False
)

test_data_iterator = batch_iterator_factory(
    torch_test,
    args.batch,
    n_clusters=args.batch_clusters,
    size_getter=(lambda x: len(x["words"])) if args.batch_clusters > 0 else None,
    shuffle=False
) if len(test_data) > 0 else None


print_log("Building network")
network = BiaffineParserNetwork(
    args,
    n_chars=len(dictionnaries["chars"]),
    n_words=len(dictionnaries["words"]),
    n_tags=len(dictionnaries["tags"]),
    n_cont_labels=len(dictionnaries["cont_labels"]),
    n_disc_labels=len(dictionnaries["disc_labels"]),
    unk_word_index=dictionnaries["words"].unk_index,
    default_lstm_init=args.default_lstm_init,
    ext_word_dict=dictionnaries["ext_words"] if "ext_words" in dictionnaries else None
)
network.to(device=args.device)

optimizer = MetaOptimizer(
    args,
    network.parameters(),
    steps_per_epoch=train_data_iterator.n_updates_per_epoch(),
    n_epochs=args.epochs
)

# load library
# -2 because we remove BOS and EOS
#max_sentence_size = max(len(sentence["words"]) for sentence in dev_data + test_data + (train_data if args.margin_loss else []))
max_sentence_size = max(len(sentence["words"]) for sentence in dev_data + test_data)
print_log("Loading cpp lib with max sent size: %i" % max_sentence_size)
disc_span_parser.load_cpp_lib(
    max_sentence_size,
    argmax_disc=True,
    n_charts=1
)

print_log("Training!\n")
best_dev_score = float("-inf")
best_dev_epoch = -1

print_log(
    "Epoch\tloss\t|"
    "\tDev uas\t\tDev las\t|"
    "\tTest uas\tTest las\t|"
    "\te time\tt time"
)
timers = pydestruct.timer.Timers()
timers.total.reset(restart=True)

loss_builder = BatchUnstructuredApproximateProbLoss(joint=not args.pipeline, reduction="mean" if args.mean_loss else "sum")
#if args.nll_loss:
#    #loss_builder = BatchUnstructuredApproximateProbLoss(joint=not args.pipeline, reduction="mean" if args.mean_loss else "sum")
#    loss_builder = BatchUnstructuredCorrectProbLoss(joint=not args.pipeline, reduction="mean" if args.mean_loss else "sum")
#elif args.margin_loss:
#    loss_builder = MarginLoss(complexity=args.complexity, ill_nested=args.ill_nested, reduction="mean" if args.mean_loss else "sum")
#else:
#    loss_builder = CorrectedBatchUnstructuredProbLoss(joint=not args.pipeline, reduction="mean" if args.mean_loss else "sum")

tag_loss_builder = nn.CrossEntropyLoss(reduction="mean" if args.mean_loss else "sum")
for epoch in range(args.epochs):
    network.train()
    epoch_loss = 0
    timers.epoch.reset(restart=True)

    for bid, inputs in enumerate(train_data_iterator):
        optimizer.zero_grad()

        # compute stuff
        span_weights, label_weights, tag_weights = network(
            inputs,
            batched=True
        )

        batch_loss = loss_builder(
            span_weights,
            label_weights,
            inputs
        )

        if tag_weights is not None:
            tag_weights = tag_weights[:, 1:]
            mask = torch.arange(0, tag_weights.shape[1], device=tag_weights.device)
            mask = mask < torch.LongTensor([len(sentence["tags"])-2 for sentence in inputs]).to(tag_weights.device).unsqueeze(1)

            gold_indices = torch.empty(tag_weights.shape[:2], dtype=torch.long, requires_grad=False, device=tag_weights.device)
            for b, sentence in enumerate(inputs):
                gold_indices[b, :len(sentence["tags"])-2] = (sentence["tags"].to(tag_weights.device)[1:-1] - 2)
            batch_loss += tag_loss_builder(tag_weights[mask], gold_indices[mask])

        # compute loss
        # optimization
        epoch_loss += batch_loss.item()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 5.0)

        optimizer.step()

    timers.epoch.stop()
    # Dev evaluation
    # We use an internal evaluator, therefore we also eval on punctuations
    # It is probably ok?
    network.eval()

    timers.dev.reset(restart=True)
    with torch.no_grad():
        dev_recall, dev_prec, dev_f1, dev_tag_acc = disc_span_parser.biaffine_eval.eval(network, dictionnaries, dev_data_iterator, pipeline=args.pipeline)
        test_recall, test_prec, test_f1, test_tag_acc = float("nan"), float("nan"), float("nan"), float("nan")
        if len(test_data) > 0:
            test_recall, test_prec, test_f1, test_tag_acc = disc_span_parser.biaffine_eval.eval(network, dictionnaries, test_data_iterator, pipeline=args.pipeline)

    timers.dev.stop()
    optimizer.eval_step(dev_f1)

    # check if dev score improved
    dev_improved = False
    if dev_f1 > best_dev_score:
        dev_improved = True
        best_dev_score = dev_f1
        best_dev_epoch = epoch
        torch.save(
            {
                "dicts": dictionnaries,
                # will be used to recreate the network, note that this can lead to privacy issue (store paths)
                # TODO: maybe use a "sub-argument pydestruct" or something like that
                "args": args,
                'model_state_dict': network.state_dict()
            },
            args.model
        )

    print_log(
        "%i\t\t%.4f\t\t%.5f\t|\t%.2f\t%.2f\t%.2f\t%.2f\t|\t|\t%.2f\t%.2f\t%.2f\t%.2f\t|\t%.2f\t%.2f\t%.2f"
        %
        (
            epoch + 1,
            epoch_loss,
            optimizer.optimizer.param_groups[0]["lr"],
            100 * dev_recall,
            100 * dev_prec,
            100 * dev_f1,
            100 * dev_tag_acc,
            100 * test_recall,
            100 * test_prec,
            100 * test_f1,
            100 * test_tag_acc,
            timers.epoch.minutes(),
            timers.dev.minutes(),
            timers.total.minutes()
        )
    )


timers.total.stop()
print_log("\nDone!")
print_log("Total training time (min): %.2f" % timers.total.minutes())
