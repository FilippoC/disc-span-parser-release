#pragma once

#include "utils/log.h"

struct InsideOutsideDiscChart
{
    const unsigned _size;
    const unsigned _n_labels;

    LogSemiring* _data_forward_labels_cont;
    LogSemiring* _data_backward_labels_cont;
    LogSemiring* _data_forward_spans_cont;
    LogSemiring* _data_backward_spans_cont;

    LogSemiring* _data_forward_labels_disc;
    LogSemiring* _data_backward_labels_disc;
    LogSemiring* _data_forward_spans_disc;
    LogSemiring* _data_backward_spans_disc;

    InsideOutsideDiscChart(unsigned size, unsigned n_labels);
    ~InsideOutsideDiscChart();

    inline unsigned offset_labeled(int i, int j, int label) const noexcept;
    inline unsigned offset_unlabeled(int i, int j) const noexcept;

    inline unsigned offset_labeled(int i, int k, int l, int j, int label) const noexcept;
    inline unsigned offset_unlabeled(int i, int k, int l, int j) const noexcept;

    inline LogSemiring& forward_labeled(int i, int j, int label) noexcept;
    inline LogSemiring& backward_labeled(int i, int j, int label) noexcept;
    inline LogSemiring& forward_unlabeled(int i, int j) noexcept;
    inline LogSemiring& backward_unlabeled(int i, int j) noexcept;

    inline LogSemiring& forward_labeled(int i, int k, int l, int j, int label) noexcept;
    inline LogSemiring& backward_labeled(int i, int k, int l, int j, int label) noexcept;
    inline LogSemiring& forward_unlabeled(int i, int k, int l, int j) noexcept;
    inline LogSemiring& backward_unlabeled(int i, int k, int l, int j) noexcept;
};

#include "chart-impl.h"