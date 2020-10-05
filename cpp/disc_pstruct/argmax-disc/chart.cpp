#include "chart.h"

#include <stdexcept>

ArgmaxDiscChart::ArgmaxDiscChart(const unsigned size) :
        _size(size + 1)
{
    auto const size_2d = _size * _size;
    auto const size_4d = _size * _size * _size * _size;

    try
    {
        _data_2d = new ArgmaxDiscChartValue[size_2d];
        _data_4d = new ArgmaxDiscChartValue[size_4d];

        // init cst values
        for (int i = 0 ; i < (int) _size ; ++i)
            for (int j = i + 1; j < (int) _size ; ++j)
            {
                (*this)(i, j).cst = ChartCstDisc(i, -1, -1, j - 1);
                for (int k = i + 1; k < j; ++k)
                    for (int l = k + 1; l < j; ++l)
                        (*this)(i, k, l, j).cst = ChartCstDisc(i, k - 1, l, j - 1);
            }

    }
    catch(std::bad_alloc&)
    {
        throw std::runtime_error("Unable to allocate memory for chart");
    }
}

ArgmaxDiscChart::~ArgmaxDiscChart()
{
    if (_size > 0)
    {
        delete[] _data_2d;
        delete[] _data_4d;
    }
}

ArgmaxCubicChart::ArgmaxCubicChart(const unsigned size, const unsigned n_labels) :
        _size(size + 1),
        _n_labels(n_labels)
{
    auto const size_2d = _size * _size;
    auto const size_3d = _size * _size * _size * n_labels;

    try
    {
        _data_top = new ArgmaxCubicChartValue[size_2d];
        _data_bottom = new ArgmaxCubicChartValue[size_3d];

        // init cst values
        for (int i = 0 ; i < (int) _size ; ++i)
            for (int j = i + 1; j < (int) _size ; ++j)
            {
                // top
                (*this)(i, j).cst = ChartCstCubic(i, j - 1, -1);
                for (int label = 0 ; label < (int) n_labels ; ++label)
                    // bottom
                    (*this)(i, j, label).cst = ChartCstCubic(i, j - 1, label);
            }
    }
    catch(std::bad_alloc&)
    {
        throw std::runtime_error("Unable to allocate memory for chart");
    }
}

ArgmaxCubicChart::~ArgmaxCubicChart()
{
    if (_size > 0)
    {
        delete[] _data_top;
        delete[] _data_bottom;
    }
}
