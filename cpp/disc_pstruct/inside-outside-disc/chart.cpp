#include <stdexcept>

#include "chart.h"

InsideOutsideDiscChart::InsideOutsideDiscChart(const unsigned size, const unsigned n_labels) :
        _size(size + 1),
        _n_labels(n_labels)
{
    auto const label_size_cont = _size * _size * n_labels;
    auto const span_size_cont = _size * _size;
    auto const label_size_disc = _size * _size *_size * _size * n_labels;
    auto const span_size_disc = _size * _size *_size * _size;

    try
    {
        _data_forward_labels_cont = new LogSemiring[label_size_cont];
        _data_backward_labels_cont = new LogSemiring[label_size_cont];
        _data_forward_spans_cont = new LogSemiring[span_size_cont];
        _data_backward_spans_cont = new LogSemiring[span_size_cont];

        _data_forward_labels_disc = new LogSemiring[label_size_disc];
        _data_backward_labels_disc = new LogSemiring[label_size_disc];
        _data_forward_spans_disc = new LogSemiring[span_size_disc];
        _data_backward_spans_disc = new LogSemiring[span_size_disc];
    }
    catch(std::bad_alloc&)
    {
        throw std::runtime_error("Unable to allocate memory for chart");
    }
}

InsideOutsideDiscChart::~InsideOutsideDiscChart()
{
    delete[] _data_forward_labels_cont;
    delete[] _data_backward_labels_cont;
    delete[] _data_forward_spans_cont;
    delete[] _data_backward_spans_cont;

    delete[] _data_forward_labels_disc;
    delete[] _data_backward_labels_disc;
    delete[] _data_forward_spans_disc;
    delete[] _data_backward_spans_disc;
}
