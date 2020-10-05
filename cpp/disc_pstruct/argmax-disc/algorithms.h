#include <stdexcept>

#include <Python.h>
#include "disc_pstruct/argmax-disc/chart.h"
#include "disc_pstruct/set.h"

PyObject* max_recall(
        ArgmaxDiscChart* chart,
        const unsigned size,
        SpanMapInt* gold,
        const unsigned complexity,
        const bool ill_nested
);

PyObject* argmax_disc_as_list(
        ArgmaxDiscChart* chart,
        unsigned size,
        unsigned n_labels,
        unsigned n_disc_labels,
        const float* weights_cont,
        const float* weights_disc,
        const float* weights_gap,
        const SpanMapInt* gold,
        const unsigned complexity,
        const bool ill_nested
);

PyObject* argmax_disc_as_sentence_ptr(
        ArgmaxDiscChart* chart,
        PyObject* py_sentence,
        PyObject* py_label_id_to_string,
        PyObject* py_disc_label_id_to_string,
        const float* weights_cont,
        const float* weights_disc,
        const float* weights_gap,
        const unsigned complexity,
        const bool ill_nested
);

const int
    RULE_COMBINE = 0,
    RULE_CREATE_GAP = 1,
    RULE_FILL_GAP = 2,
    RULE_WRAPPING = 3,
    RULE_COMBINE_KEEP_LEFT = 4,
    RULE_COMBINE_KEEP_RIGHT = 5,
    RULE_COMBINE_SHRINK_LEFT = 6,
    RULE_COMBINE_SHRINK_RIGHT = 7,
    RULE_IL_NO_GAP = 8,
    RULE_IL_GAP_CENTER = 9,
    RULE_IL_GAP_LEFT = 10,
    RULE_IL_GAP_RIGHT = 11
    ;


template <class Op>
inline const ArgmaxDiscChartValue& argmax_disc(
        ArgmaxDiscChart& chart,
        const unsigned size, Op Weights,
        const unsigned complexity=6,
        const bool ill_nested=true
)
{
    /*
     * WARNING!
     * The indexing in this algorithm is weird:
     * - For chart items we use interstice as indices to simplify the algorithm
     * - For weight, the indices are similar to other algorithms
     * Note that for discontinuous constituent spans, we give the indices of the first and last
     */

    // +1 because we use interstices, but this is transparent for the user,
    // the +1 is done in the chart constructor tooo
    if (size + 1 > chart._size)
        throw std::runtime_error("Sentence too long");

    for (int i = 0u ; i < (int) size ; ++i)
        chart(i, i + 1).axiom(Weights(i, -1, -1, i));

    for (int length = 2u; length <= (int) size; ++length)
    {
        for (int i = 0u; i <= (int) size - length; ++i)
        {
            int j = i + length;

            // we first do discontinuous constituent
            // as they can be used to build [i, -, -, j]
            if (complexity >= 4)
            {
                for (int gap = length - 2 ; gap >= 1 ; --gap)
                {
                    for (int k = i + 1 ; k < j - gap ; ++k)
                    {
                        int l = k + gap;
                                                                         // init discontinuous_constituent
                        auto &discontinuous_constituent = chart(i, k, l, j);
                        discontinuous_constituent.reset(Weights(i, k - 1, l, j - 1), true);

                        // Create gap
                        // [i, -, -, k] + [l, -, -, j] => [i, k, l, j]
                        discontinuous_constituent.deduce(chart(i, k), chart(l, j), RULE_CREATE_GAP);

                        if (complexity >= 5)
                        {
                            // Binary move with gap on the right
                            // [i, -, -, m] + [m, k, l, j] -> [i, k, l, j]
                            for (int m = i + 1; m < k; ++m)
                                discontinuous_constituent.deduce(chart(i, m), chart(m, k, l, j), RULE_COMBINE_KEEP_RIGHT);

                            // Binary move with gap on the left
                            // [i, k, l, m] + [m, j] -> [i, k, l, j]
                            for (int m = l + 1; m < j; ++m)
                                discontinuous_constituent.deduce(chart(i, k, l, m), chart(m, j), RULE_COMBINE_KEEP_LEFT);

                            // Fill gap left
                            // [i, m, l, j] + [m, k] -> [i, k, l, j]
                            for (int m = i + 1; m < k; ++m)
                                discontinuous_constituent.deduce(chart(i, m, l, j), chart(m, k), RULE_COMBINE_SHRINK_LEFT);

                            // Fill gap right
                            // [i, k, m, j] + [l, m] -> [i, k, l, j]
                            for (int m = l + 1; m < j; ++m)
                                discontinuous_constituent.deduce(chart(i, k, m, j), chart(l, m), RULE_COMBINE_SHRINK_RIGHT);
                        }

                        if (complexity >= 6)
                        {
                            // Fill gap both sides
                            // [i, m, n, j] + [m, k, l, n] -> [i, k, l, j]
                            for (int m = i + 1; m < k; ++m)
                                for (int n = l + 1; n < j; ++n)
                                    discontinuous_constituent.deduce(chart(i, m, n, j),
                                                                     chart(m, k, l, n), RULE_WRAPPING);

                            if (ill_nested)
                            {
                                // ill-nested construction of discontinuous constituent when both have a gap
                                // [i, m, l, n] + [m, k, n, j] -> [i, k, l, j]
                                for (int m = i + 1; m < k; ++m)
                                    for (int n = l + 1; n < j; ++n)
                                        discontinuous_constituent.deduce(chart(i, m, l, n),
                                                                         chart(m, k, n, j), RULE_IL_GAP_CENTER);

                                // ill-nested construction of discontinuous constituent when we fill the gap in the first part
                                // [i, m, n, k] + [m, n, l, j] -> [i, k, l, j]
                                for (int m = i + 1; m < k; ++m)
                                    for (int n = m + 1; n < k; ++n)
                                        discontinuous_constituent.deduce(chart(i, m, n, k),
                                                                         chart(m, n, l, j), RULE_IL_GAP_RIGHT);

                                // ill-nested construction of discontinuous constituent when we fill the gap in the second part
                                // [i, k, m, n] + [l, m, n, j] -> [i, k, l, j]
                                for (int m = l + 1; m < j; ++m)
                                    for (int n = m + 1; n < j; ++n)
                                        discontinuous_constituent.deduce(chart(i, k, m, n),
                                                                         chart(l, m, n, j), RULE_IL_GAP_LEFT);
                            }
                        }
                    }
                }
            }

            // we now turn to continuous consequent
            auto& continuous_consequent = chart(i, j);
            continuous_consequent.reset(Weights(i, -1, -1, j - 1), true);

            // binary move
            // [i, -, -, m] + [m, -, -, j] -> [i, -, -, j]
            for (int m = i + 1 ; m < j ; ++m)
                continuous_consequent.deduce(chart(i, m), chart(m, j), RULE_COMBINE);

            if (complexity >= 4)
            {
                // fill gap
                // [i, m, n, j] + [m, -, -, n] -> [i, -, -, j]
                for (int m = i + 1; m < j; ++m)
                    for (int n = m + 1; n < j; ++n)
                        continuous_consequent.deduce(chart(i, m, n, j), chart(m, n), RULE_FILL_GAP);
            }

            if (ill_nested && complexity >= 5)
            {
                // ill-nested construction of continuous constituent
                // [i, m, n, o] + [m, n, o, j] -> [i, -, -, j]
                for (int m = i + 1; m < j; ++m)
                    for (int n = m + 1; n < j; ++n)
                        for (int o = n + 1; o < j; ++o)
                            continuous_consequent.deduce(chart(i, m, n, o), chart(m, n, o, j), RULE_IL_NO_GAP);
            }
        }
    }

    return chart(0u, size);
}

template <class Op>
std::pair<float, int> max_weight_label_pair(
        Op get_weight,
        const unsigned size,
        const unsigned n_labels,
        const unsigned n_disc_labels,
        const float* const weights_cont, const float* const weights_disc, const float* const weights_gap,
        const int i, const int k, const int l, const int j,
        const SpanMapInt* gold = nullptr
)
{
    int gold_label = -1;
    if (gold != nullptr)
    {
        auto it = gold->find(std::make_tuple(i, k, l, j));
        if (it != gold->end())
            gold_label = it->second;
    }

    if (k >= 0)
    {
        float max_weight = -std::numeric_limits<float>::infinity();
        int max_label = -1;
        for (int label = 0 ; label < (int) n_disc_labels ; ++label)
        {
            const float weight =
                    get_weight(weights_disc, size, n_disc_labels, i, j, label)
                    +
                    get_weight(weights_gap, size, n_disc_labels, k + 1, l - 1, label)
                    +
                    (gold_label == -1 || gold_label == label ? 0. : 1.)
            ;
            if (weight > max_weight)
            {
                max_weight = weight;
                max_label = label;
            }
        }
        return {max_weight, max_label};
    }
    else
    {
        float max_weight = -std::numeric_limits<float>::infinity();
        int max_label = -1;
        for (int label = 0 ; label < (int) n_labels ; ++label)
        {
            const float weight =
                    get_weight(weights_cont, size, n_labels, i, j, label)
                    +
                    (gold_label == -1 || gold_label == label ? 0. : 1.)
                    ;

            if (weight > max_weight)
            {
                max_weight = weight;
                max_label = label;
            }
        }
        return {max_weight, max_label};
    }
}



PyObject* argmax_cubic_as_sentence_ptr(
        ArgmaxCubicChart* chart,
        PyObject* py_sentence,
        PyObject* py_label_id_to_string,
        PyObject* py_disc_label_id_to_string,
        const float* weights_cont,
        const float* weights_disc,
        const float* weights_gap
);

template <class Op1, class Op2, class Op3>
inline const ArgmaxCubicChartValue& argmax_cubic(
        ArgmaxCubicChart& chart,
        const unsigned size,
        const unsigned n_labels,
        Op1 ContWeights,
        Op2 DiscSpanWeights,
        Op3 DiscGapWeights
)
{
    /*
     * WARNING!
     * The indexing in this algorithm is weird:
     * - For chart items we use interstice as indices to simplify the algorithm
     * - For weight, the indices are similar to other algorithms
     * Note that for discontinuous constituent spans, we give the indices of the first and last
     */

    // +1 because we use interstices, but this is transparent for the user,
    // the +1 is done in the chart constructor tooo
    if (size + 1 > chart._size)
        throw std::runtime_error("Sentence too long");
    if (n_labels > chart._n_labels)
        throw std::runtime_error("Too many labels for the chart");

    for (int i = 0u ; i < (int) size ; ++i)
        chart(i, i + 1).axiom(ContWeights(i, i));

    for (int length = 2u; length <= (int) size; ++length)
    {
        for (int i = 0u; i <= (int) size - length; ++i)
        {
            int j = i + length;

            auto& top_consequent = chart(i, j);
            top_consequent.reset(ContWeights(i, j - 1), true);

            // binary move
            // [top, i, m] + [top, m, j] -> [top, i, j]
            for (int m = i + 1 ; m < j ; ++m)
                top_consequent.deduce(chart(i, m), chart(m, j), RULE_COMBINE);

            // create top from disc constituent
            if (length >= 3)
            {
                for (int m = i + 2 ; m < j ; ++m)
                {
                    for (int label = 0 ; label < (int) n_labels ; ++label)
                    {
                        const float outer_score = DiscSpanWeights(i, j - 1, label);
                        top_consequent.deduce(chart(i, m, label), chart(m, j), RULE_COMBINE, outer_score);
                    }
                }
            }

            // create bottom
            for (int label = 0 ; label < (int) n_labels ; ++label)
            {
                auto& bottom_consequent = chart(i, j, label);
                bottom_consequent.reset(-std::numeric_limits<float>::infinity(), false);

                for (int m = i + 1 ; m < j ; ++m)
                {
                    const float gap_score = DiscGapWeights(m - 1, j, label);
                    bottom_consequent.deduce(chart(i, m), chart(m, j), RULE_COMBINE, gap_score, true);
                }
            }
        }
    }

    return chart(0u, size);
}