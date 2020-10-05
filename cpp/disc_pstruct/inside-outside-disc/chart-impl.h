#pragma once

unsigned InsideOutsideDiscChart::offset_labeled(const int i, const int j, const int label) const noexcept
{
    return i * _size * _n_labels + j * _n_labels + label;
}

unsigned InsideOutsideDiscChart::offset_unlabeled(const int i, const int j) const noexcept
{
    return i * _size + j;
}

unsigned InsideOutsideDiscChart::offset_labeled(const int i, const int k, const int l, const int j, const int label) const noexcept
{
    return i * _size  * _size * _size * _n_labels + j * _size  * _size * _n_labels  + k * _size * _n_labels + l * _n_labels + label;
}

unsigned InsideOutsideDiscChart::offset_unlabeled(const int i, const int k, const int l, const int j) const noexcept
{
    return i * _size  * _size  * _size + j * _size  * _size  + k * _size + l;
}

LogSemiring& InsideOutsideDiscChart::forward_labeled(const int i, const int j, const int label) noexcept
{
    return _data_forward_labels_cont[offset_labeled(i, j, label)];
}

LogSemiring& InsideOutsideDiscChart::backward_labeled(const int i, const int j, const int label) noexcept
{
    return _data_backward_labels_cont[offset_labeled(i, j, label)];
}

LogSemiring& InsideOutsideDiscChart::forward_unlabeled(const int i, const int j) noexcept
{
    return _data_forward_spans_cont[offset_unlabeled(i, j)];
}

LogSemiring& InsideOutsideDiscChart::backward_unlabeled(const int i, const int j) noexcept
{
    return _data_backward_spans_cont[offset_unlabeled(i, j)];
}

LogSemiring& InsideOutsideDiscChart::forward_labeled(const int i, const int k, const int l, const int j, const int label) noexcept
{
    return _data_forward_labels_disc[offset_labeled(i, k, l, j, label)];
}

LogSemiring& InsideOutsideDiscChart::backward_labeled(const int i, const int k, const int l, const int j, const int label) noexcept
{
    return _data_backward_labels_disc[offset_labeled(i, k, l, j, label)];
}

LogSemiring& InsideOutsideDiscChart::forward_unlabeled(const int i, const int k, const int l, const int j) noexcept
{
    return _data_forward_spans_disc[offset_unlabeled(i, k, l, j)];
}

LogSemiring& InsideOutsideDiscChart::backward_unlabeled(const int i, const int k, const int l, const int j) noexcept
{
    return _data_backward_spans_disc[offset_unlabeled(i, k, l, j)];
}