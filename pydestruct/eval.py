from __future__ import print_function
import pydestruct.data.ptb
import sys

# TODO: There are other common pre-processing steps that should be easy to implement.
#       For example, PRT and ADVP are usually considered as equivalent for evaluation
class ConstituentEvaluator:
    def __init__(self, split_unary_chains=True, sep="_"):
        self.split_unary_chains = split_unary_chains
        self.sep = sep
        self.reset()

    def reset(self):
        self.sum_recall = 0
        self.norm_recall = 0
        self.sum_precicion = 0
        self.norm_precision = 0
        self.n_exact = 0
        self.n_sentences = 0

    # Python reimplementation of https://github.com/stanfordnlp/CoreNLP/blob/39a3a247fd757971842e48abf42a31a44b668a4c/src/edu/stanford/nlp/parser/metrics/AbstractEval.java#L131
    def update(self, gold_cst, pred_cst):
        #print(gold_cst, file=sys.stderr)
        #print(pred_cst, file=sys.stderr)
        self.n_sentences += 1

        if self.split_unary_chains:
            gold_cst = pydestruct.data.ptb.split_unary_chains(gold_cst, self.sep)
            pred_cst = pydestruct.data.ptb.split_unary_chains(pred_cst, self.sep)

        common_cst = gold_cst.intersection(pred_cst)
        precision = len(common_cst) / len(pred_cst) if len(pred_cst) > 0 else 0
        recall = len(common_cst) / len(gold_cst) if len(gold_cst) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision > 0 and recall > 0 else 0

        self.sum_precicion += precision * len(pred_cst)
        self.norm_precision += len(pred_cst)
        self.sum_recall += recall * len(gold_cst)
        self.norm_recall += len(gold_cst)

        if f1 > 0.9999:
            self.n_exact += 1

    def recall(self):
        return self.sum_recall / self.norm_recall if self.norm_recall > 0 else 0

    def precision(self):
        return self.sum_precicion / self.norm_precision if self.norm_precision > 0 else 0

    def f1(self):
        prec = self.precision()
        rec = self.recall()
        return (2.0 * (prec * rec) / (prec + rec)) if prec > 0 and rec > 0 else 0

    def exact_match(self):
        return (self.n_exact / self.n_sentences) if self.n_sentences != 0 else 0

    def print(self, file=sys.stderr):
        print("precision: %.2f" % (100 * self.precision()), file=file)
        print("recall: %.2f" % (100 * self.recall()), file=file)
        print("f-measure: %.2f" % (100 * self.f1()), file=file)
        print("exact match: %.2f" % (100 * self.exact_match()), file=file)
