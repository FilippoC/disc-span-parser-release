inline
unsigned ArgmaxDiscChart::offset(const unsigned i, const unsigned j) const noexcept
{
    return i * _size + j;
}

inline
unsigned ArgmaxDiscChart::offset(const unsigned i, const unsigned k, const unsigned l, const unsigned j) const noexcept
{
    return i * _size * _size * _size + k * _size * _size + l * _size + j;
}

inline
ArgmaxDiscChartValue& ArgmaxDiscChart::operator()(const unsigned i, const unsigned j) noexcept
{
    return _data_2d[offset(i, j)];
}

inline
ArgmaxDiscChartValue& ArgmaxDiscChart::operator()(const unsigned i, const unsigned k, const unsigned l, const unsigned j) noexcept
{
    return _data_4d[offset(i, k, l, j)];
}

inline
unsigned ArgmaxCubicChart::offset(const unsigned i, const unsigned j) const noexcept
{
    return i * _size + j;
}

inline
unsigned ArgmaxCubicChart::offset(const unsigned i, const unsigned j, const unsigned label) const noexcept
{
    return i * _size * _n_labels + j * _n_labels + label;
}

inline
ArgmaxCubicChartValue& ArgmaxCubicChart::operator()(const unsigned i, const unsigned j) noexcept
{
    return _data_top[offset(i, j)];
}

inline
ArgmaxCubicChartValue& ArgmaxCubicChart::operator()(const unsigned i, const unsigned j, const unsigned label) noexcept
{
    return _data_bottom[offset(i, j, label)];
}
