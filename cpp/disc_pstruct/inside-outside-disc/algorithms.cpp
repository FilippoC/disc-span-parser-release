
/*
InsideOutsideDiscChart* new_InsideOutsideDiscChart(unsigned size, unsigned n_labels)
{
    return new InsideOutsideDiscChart(size, n_labels);
}

float log_partition_and_marginals_disc(
        InsideOutsideDiscChart* _chart,
        const unsigned size,
        const unsigned n_labels,
        const float* const cont_weights,
        const float* const disc_weights,
        const float* const gap_weights,
        float* cont_marginals,
        float* disc_marginals,
        float* gap_marginals,
        const bool fencepost
)
{
    if (fencepost)
        return log_partition_and_marginals_disc(
                *_chart,
                size,
                n_labels,
                [&] (const int i, const int j, const int label) -> float
                {
                    return torch_get_labeled_cst_weight(cont_weights, size, n_labels, i, j, label);
                },
                [&] (const int i, const int k, const int l, const int j, const int label) -> float
                {
                    return
                            torch_get_labeled_cst_weight(disc_weights, size, n_labels, i, j, label)
                            +
                            torch_get_labeled_cst_weight(gap_weights, size, n_labels, k, l, label)
                            ;
                },
                [&] (const int i, const int j, const int label) -> float&
                {
                    return torch_get_labeled_cst_weight(cont_marginals, size, n_labels, i, j, label);
                },
                [&] (const int i, const int k, const int l, const int j, const int label, const float value)
                {
                    torch_get_labeled_cst_weight(disc_marginals, size, n_labels, i, j, label) += value;
                    torch_get_labeled_cst_weight(gap_marginals, size, n_labels, k, l, label) += value;
                }
        );
    else
        return log_partition_and_marginals_disc(
                *_chart,
                size,
                n_labels,
                [&] (const int i, const int j, const int label) -> float
                {
                    return torch_get_labeled_cst_weight(cont_weights, size, n_labels, i, j, label);
                },
                [&] (const int i, const int k, const int l, const int j, const int label) -> float
                {
                    return
                            torch_get_labeled_cst_weight(disc_weights, size, n_labels, i, j, label)
                            +
                            torch_get_labeled_cst_weight(gap_weights, size, n_labels, k, l, label)
                            ;
                },
                [&] (const int i, const int j, const int label) -> float&
                {
                    return torch_get_labeled_cst_weight(cont_marginals, size, n_labels, i, j, label);
                },
                [&] (const int i, const int k, const int l, const int j, const int label, const float value)
                {
                    torch_get_labeled_cst_weight(disc_marginals, size, n_labels, i, j, label) += value;
                    torch_get_labeled_cst_weight(gap_marginals, size, n_labels, k, l, label) += value;
                }
        );
}
 */