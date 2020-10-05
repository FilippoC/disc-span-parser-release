import disc_span_parser
import pydestruct.eval

def eval(network, dictionnaries, data_iterator, pipeline, complexity=6, ill_nested=True):
    evaluator = pydestruct.eval.ConstituentEvaluator(sep="/")

    for inputs in data_iterator:
        # compute stuff
        span_weights, label_weights, tag_weights = network(
            inputs,
            batched=True
        )

        n_tags = 0.
        n_correct_tags = 0.
        for b, sentence in enumerate(inputs):
            n_words = len(inputs[b]["words"]) - 2
            cont_spans = span_weights["cont"][b, 1:n_words+1,  1:n_words+1]
            cont_labels = label_weights["cont"][b, 1:n_words+1,  1:n_words+1]
            disc_spans = span_weights["disc"][b, 1:n_words+1,  1:n_words+1]
            disc_labels = label_weights["disc"][b, 1:n_words+1,  1:n_words+1]
            gap_spans = span_weights["gap"][b, 1:n_words+1,  1:n_words+1]
            gap_labels = label_weights["gap"][b, 1:n_words+1,  1:n_words+1]

            if tag_weights is not None:
                n_tags += n_words
                _, pred_tags = tag_weights[b, 1:n_words + 1].max(dim=1)
                n_correct_tags += (pred_tags == (inputs[b]["tags"].to(pred_tags.device)[1:-1] -2)).sum().item()

            if pipeline:
                pred_cst = disc_span_parser.argmax_as_list_parallel(
                    list([cont_spans]),
                    list([disc_spans]),
                    list([gap_spans]),
                    None,
                    complexity,
                    ill_nested
                )[0]

                _, cont_labels = cont_labels.max(2)

                # retrieve predicted labels
                pred = set()
                n_disc_labels = disc_labels.shape[2]
                for _, i, k, l, j in pred_cst:
                    if k < 0:
                        label = dictionnaries["cont_labels"].id_to_word(cont_labels[i, j].item())
                    else:
                        _, label = max(
                            (disc_labels[i, j, label].item() + gap_labels[k, l, label].item(), label)
                            for label in range(n_disc_labels)
                        )
                        label = dictionnaries["disc_labels"].id_to_word(label)
                    pred.add((label, i, k, l, j))

            else:
                pred_cst = disc_span_parser.argmax_as_list_parallel(
                    list([cont_spans + cont_labels]),
                    list([disc_spans + disc_labels]),
                    list([gap_spans + gap_labels]),
                    None,
                    complexity,
                    ill_nested
                )[0]

                # retrieve predicted labels
                pred = set(
                    (dictionnaries["cont_labels"].id_to_word(label) if k < 0 else dictionnaries["disc_labels"].id_to_word(label), i, k, l, j)
                    for label, i, k, l, j in pred_cst
                )
            # add to the evaluator
            evaluator.update(sentence["constituents"], pred)

    recall = evaluator.recall()
    prec = evaluator.precision()
    f1 = evaluator.f1()
    tag_acc = (n_correct_tags / n_tags) if n_tags > 0 else float("nan")

    return recall, prec, f1, tag_acc
