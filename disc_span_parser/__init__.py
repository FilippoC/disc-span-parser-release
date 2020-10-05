import pydestruct.functional
import torch.nn as nn
import torch
import cpp_disc_span_parser

LIB_PARALLEL = False
ARGMAX_DISC_CHART = None
ARGMAX_CUBIC_CHART = None

def load_cpp_lib_cubic(max_size, n_labels):
    global ARGMAX_CUBIC_CHART
    if max_size <= 0:
        raise RuntimeError("The chart size must be positive")

    ARGMAX_CUBIC_CHART = cpp_disc_span_parser.new_ArgmaxCubicChart(max_size, n_labels)

def load_cpp_lib(max_size, argmax_disc=False, io_disc=False, n_charts=1):
    global ARGMAX_DISC_CHART, LIB_PARALLEL
    if max_size <= 0:
        raise RuntimeError("The chart size must be positive")

    if not (argmax_disc or io_disc):
        raise RuntimeError("No chart where requested")

    if argmax_disc:
        if n_charts > 1:
            ARGMAX_DISC_CHART = cpp_disc_span_parser.new_ArgmaxDiscChart_parallel(max_size, n_charts)
            LIB_PARALLEL = True
        else:
            ARGMAX_DISC_CHART = cpp_disc_span_parser.new_ArgmaxDiscChart(max_size)
            LIB_PARALLEL = False
    if io_disc:
        raise NotImplementedError()


def _check_labeled(input):
    if not input.is_contiguous():
        raise RuntimeError("Tensor memory is not contiguous")
    if not isinstance(input, torch.FloatTensor):
        raise RuntimeError("Dynamic programming functions can only be called weights CPU float 32 tensors weights.")
    if len(input.size()) != 3:
        raise RuntimeError("Labeled weights must be tensors of size 3")
    if input.size()[0] != input.size()[1]:
        raise RuntimeError("The two first dimensions should be equal")


def argmax_as_list(cont_weights, disc_weights, gap_weights, complexity, ill_nested):
    if LIB_PARALLEL:
        raise RuntimeError("Cannot use this in parallel mode!")
    cont_weights = cont_weights.contiguous().cpu()
    disc_weights = disc_weights.contiguous().cpu()
    gap_weights = gap_weights.contiguous().cpu()
    pred_cst = cpp_disc_span_parser.argmax_disc_as_list(
        ARGMAX_DISC_CHART,
        cont_weights.shape[0],
        cont_weights.shape[2],
        cont_weights.data_ptr(),
        disc_weights.data_ptr(),
        gap_weights.data_ptr(),
        complexity,
        ill_nested
    )
    return pred_cst


def argmax_as_list_parallel(cont_weights_list, disc_weights_list, gap_weights_list, gold_spans, complexity, ill_nested):
    with torch.no_grad():
        a = list(w.contiguous().cpu() for w in cont_weights_list)
        b = list(w.contiguous().cpu() for w in disc_weights_list)
        c = list(w.contiguous().cpu() for w in gap_weights_list)
        a_ptr = [w.data_ptr() for w in a]
        b_ptr = [w.data_ptr() for w in b]
        c_ptr = [w.data_ptr() for w in c]
        if LIB_PARALLEL:
            return cpp_disc_span_parser.argmax_disc_as_list_parallel(
                ARGMAX_DISC_CHART,
                list(w.shape[0] for w in cont_weights_list),
                list(w.shape[2] for w in cont_weights_list),
                a_ptr, b_ptr, c_ptr,
                list(gold_spans) if gold_spans is not None else None,
                complexity, ill_nested
            )
        else:
            ret = list()
            for i in range(len(cont_weights_list)):
                if gold_spans is not None:
                    ret.append(cpp_disc_span_parser.argmax_disc_as_list_with_gold_spans(
                        ARGMAX_DISC_CHART,
                        cont_weights_list[i].shape[0],
                        cont_weights_list[i].shape[2],
                        disc_weights_list[i].shape[2],
                        a_ptr[i],
                        b_ptr[i],
                        c_ptr[i],
                        gold_spans[i],
                        complexity,
                        ill_nested
                    ))
                else:
                    ret.append(cpp_disc_span_parser.argmax_disc_as_list(
                        ARGMAX_DISC_CHART,
                        cont_weights_list[i].shape[0],
                        cont_weights_list[i].shape[2],
                        disc_weights_list[i].shape[2],
                        a_ptr[i],
                        b_ptr[i],
                        c_ptr[i],
                        complexity,
                        ill_nested
                    ))
            return ret
