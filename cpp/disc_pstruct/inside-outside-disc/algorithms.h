#include <stdexcept>

#include "disc_pstruct/inside-outside-disc/chart.h"
#include "utils/torch-indexing.h"


template <class T, class U, class V, class W>
float log_partition_and_marginals_disc(
        InsideOutsideDiscChart& chart,
        const unsigned size,
        const unsigned n_labels,
        T get_cont_weight,
        U get_disc_weight,
        V get_cont_marginal,
        W add_disc_marginal
)
{
    const unsigned complexity = 6;
    const bool ill_nested = true;
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

    for (int i = 0u; i < (int) size; ++i)
    {
        chart.forward_unlabeled(i, i + 1) = LogSemiring::null();
        for (int label = 0; label < (int) n_labels; ++label)
        {
            const auto v = LogSemiring::from_log(get_cont_weight(i, i, label));
            chart.forward_labeled(i, i + 1, label) = v;
            chart.forward_unlabeled(i, i + 1) += v;
        }
        // for the null label for implicit binarization
        //chart.forward_unlabeled(i, i + 1) += LogSemiring::one();
    }

    for (int length = 2u; length <= (int) size; ++length)
    {
        for (int i = 0u; i <= (int) size - length; ++i)
        {
            int j = i + length;
            chart.forward_unlabeled(i, j) = LogSemiring::null();

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
                        chart.forward_unlabeled(i, k, l, j) = LogSemiring::null();
                        for (int label = 0; label < (int) n_labels; ++label)
                        {
                            chart.forward_labeled(i, k, l, j, label) = LogSemiring::null();

                            const LogSemiring disc_weight = LogSemiring::from_log(get_disc_weight(i, k, l-1, j-1, label));
                            LogSemiring &forward_labeled_disc = chart.forward_labeled(i, k, l, j, label);
                            LogSemiring &forward_unlabeled_disc = chart.forward_unlabeled(i, k, l, j);

                            // Create gap
                            // [i, -, -, k] + [l, -, -, j] => [i, k, l, j]
                            const LogSemiring v =
                                    chart.forward_unlabeled(i, k)
                                    * chart.forward_unlabeled(l, j)
                                    * disc_weight;
                            forward_labeled_disc += v;
                            forward_unlabeled_disc += v;

                            if (complexity >= 5)
                            {
                                // Binary move with gap on the right
                                // [i, -, -, m] + [m, k, l, j] -> [i, k, l, j]
                                for (int m = i + 1; m < k; ++m)
                                {
                                    const LogSemiring v =
                                            chart.forward_unlabeled(i, m)
                                            * chart.forward_unlabeled(m, k, l, j)
                                            * disc_weight;
                                    forward_labeled_disc += v;
                                    forward_unlabeled_disc += v;
                                }

                                // Binary move with gap on the left
                                // [i, k, l, m] + [m, j] -> [i, k, l, j]
                                for (int m = l + 1; m < j; ++m)
                                {
                                    const LogSemiring v =
                                            chart.forward_unlabeled(i, k, l, m)
                                            * chart.forward_unlabeled(m, j)
                                            * disc_weight;
                                    forward_labeled_disc += v;
                                    forward_unlabeled_disc += v;
                                }

                                // Fill gap left
                                // [i, m, l, j] + [m, k] -> [i, k, l, j]
                                for (int m = i + 1; m < k; ++m)
                                {
                                    //std::cerr << "[" << i <<", "<<m<<", "<<l<<", "<<j<<"] + ["<<m<<", "<<k<<"] -> ["<<i<<", "<<k<<", "<<l<<", "<<j<<"] "<< "\n";
                                    const LogSemiring v =
                                            chart.forward_unlabeled(i, m, l, j)
                                            * chart.forward_unlabeled(m, k)
                                            * disc_weight;
                                    forward_labeled_disc += v;
                                    forward_unlabeled_disc += v;
                                }

                                // Fill gap right
                                // [i, k, m, j] + [l, m] -> [i, k, l, j]
                                for (int m = l + 1; m < j; ++m)
                                {
                                    const LogSemiring v =
                                            chart.forward_unlabeled(i, k, m, j)
                                            * chart.forward_unlabeled(l, m)
                                            * disc_weight;
                                    forward_labeled_disc += v;
                                    forward_unlabeled_disc += v;
                                }
                            }

                            if (complexity >= 6)
                            {
                                // Fill gap both sides
                                // [i, m, n, j] + [m, k, l, n] -> [i, k, l, j]

                                for (int m = i + 1; m < k; ++m)
                                    for (int n = l + 1; n < j; ++n)
                                    {
                                        const LogSemiring v =
                                                chart.forward_unlabeled(i, m, n, j)
                                                * chart.forward_unlabeled(m, k, l, n)
                                                * disc_weight;
                                        forward_labeled_disc += v;
                                        forward_unlabeled_disc += v;
                                    }

                                if (ill_nested)
                                {
                                    // ill-nested construction of discontinuous constituent when both have a gap
                                    // [i, m, l, n] + [m, k, n, j] -> [i, k, l, j]
                                    for (int m = i + 1; m < k; ++m)
                                        for (int n = l + 1; n < j; ++n)
                                        {
                                            const LogSemiring v =
                                                    chart.forward_unlabeled(i, m, l, n)
                                                    * chart.forward_unlabeled(m, k, n, j)
                                                    * disc_weight;
                                            forward_labeled_disc += v;
                                            forward_unlabeled_disc += v;
                                        }

                                    // ill-nested construction of discontinuous constituent when we fill the gap in the first part
                                    // [i, m, n, k] + [m, n, l, j] -> [i, k, l, j]
                                    for (int m = i + 1; m < k; ++m)
                                        for (int n = m + 1; n < k; ++n)
                                        {
                                            const LogSemiring v =
                                                    chart.forward_unlabeled(i, m, n, k)
                                                    * chart.forward_unlabeled(m, n, l, j)
                                                    * disc_weight;
                                            forward_labeled_disc += v;
                                            forward_unlabeled_disc += v;
                                        }

                                    // ill-nested construction of discontinuous constituent when we fill the gap in the second part
                                    // [i, k, m, n] + [l, m, n, j] -> [i, k, l, j]
                                    for (int m = l + 1; m < j; ++m)
                                        for (int n = m + 1; n < j; ++n)
                                        {
                                            const LogSemiring v =
                                                    chart.forward_unlabeled(i, k, m, n)
                                                    * chart.forward_unlabeled(l, m, n, j)
                                                    * disc_weight;
                                            forward_labeled_disc += v;
                                            forward_unlabeled_disc += v;
                                        }
                                }
                            }
                        }
                    }
                }
            }

            // we now turn to continuous consequent
            chart.forward_unlabeled(i, j) = LogSemiring::null();
            LogSemiring &forward_unlabeled_cont = chart.forward_unlabeled(i, j);

            // binary move
            // [i, -, -, m] + [m, -, -, j] -> [i, -, -, j]
            for (int label = 0; label < (int) n_labels; ++label)
            {
                chart.forward_labeled(i, j, label) = LogSemiring::null();

                LogSemiring &forward_labeled_cont = chart.forward_labeled(i, j, label);
                const LogSemiring cont_weight = LogSemiring::from_log(get_cont_weight(i, j-1, label));

                for (int m = i + 1; m < j; ++m)
                {
                    const LogSemiring v =
                            chart.forward_unlabeled(i, m)
                            * chart.forward_unlabeled(m, j)
                            * cont_weight;
                    forward_labeled_cont += v;
                    forward_unlabeled_cont += v;
                }

                if (complexity >= 4)
                {
                    // fill gap
                    // [i, m, n, j] + [m, -, -, n] -> [i, -, -, j]
                    for (int m = i + 1; m < j; ++m)
                        for (int n = m + 1; n < j; ++n)
                        {
                            const LogSemiring v =
                                    chart.forward_unlabeled(i, m, n, j)
                                    * chart.forward_unlabeled(m, n)
                                    * cont_weight;
                            forward_labeled_cont += v;
                            forward_unlabeled_cont += v;
                        }
                }

                if (ill_nested && complexity >= 5)
                {
                    // ill-nested construction of continuous constituent
                    // [i, m, n, o] + [m, n, o, j] -> [i, -, -, j]
                    for (int m = i + 1; m < j; ++m)
                        for (int n = m + 1; n < j; ++n)
                            for (int o = n + 1; o < j; ++o)
                            {
                                const LogSemiring v =
                                        chart.forward_unlabeled(i, m, n, o)
                                        * chart.forward_unlabeled(m, n, o, j)
                                        * cont_weight;
                                forward_labeled_cont += v;
                                forward_unlabeled_cont += v;
                            }
                }
            }
        }
    }

    // init these values here
    for (int i = 0; i <= (int) size; ++i)
        for (int j = i+1; j <= (int) size; ++j)
        {
            chart.backward_unlabeled(i, j) = LogSemiring::null();
            for (int label = 0; label < (int) n_labels; ++label)
                chart.backward_labeled(i, j, label) = LogSemiring::null();

            for (int k = i + 1; k < j; ++k)
                for (int l = k + 1; l < j; ++l)
                {
                    chart.backward_unlabeled(i, k, l, j) = LogSemiring::null();
                    for (int label = 0; label < (int) n_labels; ++label)
                        chart.backward_labeled(i, k, l, j, label) = LogSemiring::null();
                }
        }

    chart.backward_unlabeled(0, (int) size) = LogSemiring::one();

    for (int length = (int) size ; length >= 1; --length)
    {
        for (int i = 0u; i + length <= (int) size ; ++i)
        {
            int j = i + length;

            // for outside we start with continuous constituent

            // binary move
            // [i, -, -, m] + [m, -, -, j] -> [i, -, -, j]
            for (int label = 0; label < (int) n_labels; ++label)
            {
                const LogSemiring cont_weight = LogSemiring::from_log(get_cont_weight(i, j-1, label));
                chart.backward_labeled(i, j, label) += chart.backward_unlabeled(i, j);

                for (int m = i + 1; m < j; ++m)
                {
                    chart.backward_unlabeled(i, m) +=
                            chart.backward_unlabeled(i, j)
                            * chart.forward_unlabeled(m, j)
                            * cont_weight;
                    chart.backward_unlabeled(m, j) +=
                            chart.backward_unlabeled(i, j)
                            * chart.forward_unlabeled(i, m)
                            * cont_weight;
                    ;
                }

                if (complexity >= 4)
                {
                    // fill gap
                    // [i, m, n, j] + [m, -, -, n] -> [i, -, -, j]
                    for (int m = i + 1; m < j; ++m)
                        for (int n = m + 1; n < j; ++n)
                        {
                            chart.backward_unlabeled(i, m, n, j) +=
                                    chart.backward_unlabeled(i, j)
                                    * chart.forward_unlabeled(m, n)
                                    * cont_weight
                                    ;

                            chart.backward_unlabeled(m, n) +=
                                    chart.backward_unlabeled(i, j)
                                    * chart.forward_unlabeled(i, m, n, j)
                                    * cont_weight
                                    ;
                        }
                }

                if (ill_nested && complexity >= 5)
                {
                    // ill-nested construction of continuous constituent
                    // [i, m, n, o] + [m, n, o, j] -> [i, -, -, j]
                    for (int m = i + 1; m < j; ++m)
                        for (int n = m + 1; n < j; ++n)
                            for (int o = n + 1; o < j; ++o)
                            {

                                chart.backward_unlabeled(i, m, n, o) +=
                                        chart.backward_unlabeled(i, j)
                                        * chart.forward_unlabeled(m, n, o, j)
                                        * cont_weight
                                        ;

                                chart.backward_unlabeled(m, n, o, j) +=
                                        chart.backward_unlabeled(i, j)
                                        * chart.forward_unlabeled(i, m, n, o)
                                        * cont_weight
                                        ;
                            }
                }
            }

            // and now discontinuous constituents
            // as they can be used to build [i, -, -, j]
            if (complexity >= 4)
            {
                for (int gap = 1 ; gap <= length - 2 ; ++gap)
                {
                    for (int k = i + 1 ; k < j - gap ; ++k)
                    {
                        int l = k + gap;

                        for (int label = 0; label < (int) n_labels; ++label)
                        {
                            const LogSemiring disc_weight = LogSemiring::from_log(get_disc_weight(i, k, l-1, j-1, label));
                            chart.backward_labeled(i, k, l, j, label) += chart.backward_unlabeled(i, k, l, j);


                            // Create gap
                            // [i, -, -, k] + [l, -, -, j] => [i, k, l, j]
                            chart.backward_unlabeled(i, k) +=
                                chart.backward_unlabeled(i, k, l, j)
                                * chart.forward_unlabeled(l, j)
                                * disc_weight
                                ;

                            chart.backward_unlabeled(l, j) +=
                                    chart.backward_unlabeled(i, k, l, j)
                                    * chart.forward_unlabeled(i, k)
                                    * disc_weight
                                    ;

                            if (complexity >= 5)
                            {
                                // Binary move with gap on the right
                                // [i, -, -, m] + [m, k, l, j] -> [i, k, l, j]
                                for (int m = i + 1; m < k; ++m)
                                {
                                    chart.backward_unlabeled(i, m) +=
                                            chart.backward_unlabeled(i, k, l, j)
                                            * chart.forward_unlabeled(m, k, l, j)
                                            * disc_weight
                                            ;
                                    chart.backward_unlabeled(m, k, l, j) +=
                                            chart.backward_unlabeled(i, k, l, j)
                                            * chart.forward_unlabeled(i, m)
                                            * disc_weight
                                            ;
                                }

                                // Binary move with gap on the left
                                // [i, k, l, m] + [m, j] -> [i, k, l, j]
                                for (int m = l + 1; m < j; ++m)
                                {
                                    chart.backward_unlabeled(i, k, l, m) +=
                                            chart.backward_unlabeled(i, k, l, j)
                                            * chart.forward_unlabeled(m, j)
                                            * disc_weight
                                            ;
                                    chart.backward_unlabeled(m, j) +=
                                            chart.backward_unlabeled(i, k, l, j)
                                            * chart.forward_unlabeled(i, k, l, m)
                                            * disc_weight
                                            ;
                                }

                                // Fill gap left
                                // [i, m, l, j] + [m, k] -> [i, k, l, j]
                                for (int m = i + 1; m < k; ++m)
                                {
                                    chart.backward_unlabeled(i, m, l, j) +=
                                            chart.backward_unlabeled(i, k, l, j)
                                            * chart.forward_unlabeled(m, k)
                                            * disc_weight
                                            ;
                                    chart.backward_unlabeled(m, k) +=
                                            chart.backward_unlabeled(i, k, l, j)
                                            * chart.forward_unlabeled(i, m, l, j)
                                            * disc_weight
                                          ;
                                }

                                // Fill gap right
                                // [i, k, m, j] + [l, m] -> [i, k, l, j]
                                for (int m = l + 1; m < j; ++m)
                                {
                                    chart.backward_unlabeled(i, k, m, j) +=
                                            chart.backward_unlabeled(i, k, l, j)
                                            * chart.forward_unlabeled(l, m)
                                            * disc_weight
                                            ;
                                    chart.backward_unlabeled(l, m) +=
                                            chart.backward_unlabeled(i, k, l, j)
                                            * chart.forward_unlabeled(i, k, m, j)
                                            * disc_weight
                                            ;
                                }
                            }

                            if (complexity >= 6)
                            {
                                // Fill gap both sides
                                // [i, m, n, j] + [m, k, l, n] -> [i, k, l, j]

                                for (int m = i + 1; m < k; ++m)
                                    for (int n = l + 1; n < j; ++n)
                                    {
                                        chart.backward_unlabeled(i, m, n, j) +=
                                                chart.backward_unlabeled(i, k, l, j)
                                                * chart.forward_unlabeled(m, k, l, n)
                                                * disc_weight;
                                        chart.backward_unlabeled(m, k, l, n) +=
                                                chart.backward_unlabeled(i, k, l, j)
                                                * chart.forward_unlabeled(i, m, n, j)
                                                * disc_weight;
                                    }

                                if (ill_nested)
                                {
                                    // ill-nested construction of discontinuous constituent when both have a gap
                                    // [i, m, l, n] + [m, k, n, j] -> [i, k, l, j]
                                    for (int m = i + 1; m < k; ++m)
                                        for (int n = l + 1; n < j; ++n)
                                        {
                                            chart.backward_unlabeled(i, m, l, n) +=
                                                    chart.backward_unlabeled(i, k, l, j)
                                                    * chart.forward_unlabeled(m, k, n, j)
                                                    * disc_weight
                                                    ;
                                            chart.backward_unlabeled(m, k, n, j) +=
                                                    chart.backward_unlabeled(i, k, l, j)
                                                    * chart.forward_unlabeled(i, m, l, n)
                                                    * disc_weight
                                                    ;
                                        }

                                    // ill-nested construction of discontinuous constituent when we fill the gap in the first part
                                    // [i, m, n, k] + [m, n, l, j] -> [i, k, l, j]
                                    for (int m = i + 1; m < k; ++m)
                                        for (int n = m + 1; n < k; ++n)
                                        {
                                            chart.backward_unlabeled(i, m, n, k) +=
                                                    chart.backward_unlabeled(i, k, l, j)
                                                    * chart.forward_unlabeled(m, n, l, j)
                                                    * disc_weight
                                                    ;
                                            chart.backward_unlabeled(m, n, l, j) +=
                                                    chart.backward_unlabeled(i, k, l, j)
                                                    * chart.forward_unlabeled(i, m, n, k)
                                                    * disc_weight
                                                    ;
                                        }

                                    // ill-nested construction of discontinuous constituent when we fill the gap in the second part
                                    // [i, k, m, n] + [l, m, n, j] -> [i, k, l, j]
                                    for (int m = l + 1; m < j; ++m)
                                        for (int n = m + 1; n < j; ++n)
                                        {
                                            chart.backward_unlabeled(i, k, m, n) +=
                                                    chart.backward_unlabeled(i, k, l, j)
                                                    * chart.forward_unlabeled(l, m, n, j)
                                                    * disc_weight
                                                    ;
                                            chart.backward_unlabeled(l, m, n, j) +=
                                                    chart.backward_unlabeled(i, k, l, j)
                                                    * chart.forward_unlabeled(i, k, m, n)
                                                    * disc_weight
                                                    ;
                                        }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    const float log_partition = chart.forward_unlabeled(0, (int) size).value;
    for (int i = 0 ; i <= (int) size ; ++i)
        for (int j = i + 1; j <= (int) size ; ++j)
            for (int label = 0 ; label < (int) n_labels ; ++label)
            {
                get_cont_marginal(i, j-1, label) = std::exp((chart.forward_labeled(i, j, label) * chart.backward_labeled(i, j, label)).value - log_partition);

                for (int k = i + 1; k < j; ++k)
                    for (int l = k + 1; l < j; ++l)
                            add_disc_marginal(
                                    i, k, l-1, j-1, label,
                                    std::exp((
                                            chart.forward_labeled(i, k, l, j, label)
                                            * chart.backward_labeled(i, k, l, j, label)
                                            ).value - log_partition)
                            );

            }

    return log_partition;
}

