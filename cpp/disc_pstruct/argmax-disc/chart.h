#pragma once

#include <tuple>
#include <limits>

#include "utils/chart-utils.h"

typedef std::tuple<int, int, int, int> ChartCstDisc;

struct ArgmaxDiscChartValue : public ArgmaxChartValueTemplate<ChartCstDisc, ArgmaxDiscChartValue>
{};

struct ArgmaxDiscChart
{
    const unsigned _size;
    ArgmaxDiscChartValue* _data_2d;
    ArgmaxDiscChartValue* _data_4d;

    ArgmaxDiscChart(const unsigned size);
    ~ArgmaxDiscChart();

    inline unsigned offset(unsigned i, unsigned j) const noexcept;
    inline unsigned offset(unsigned i, unsigned k, unsigned l, unsigned j) const noexcept;
    inline ArgmaxDiscChartValue& operator()(const unsigned i, const unsigned j) noexcept;
    inline ArgmaxDiscChartValue& operator()(const unsigned i, unsigned k, unsigned l, const unsigned j) noexcept;
};


// left, right, label
typedef std::tuple<int, int, int> ChartCstCubic;

struct ArgmaxCubicChartValue : public ArgmaxChartValueTemplate<ChartCstCubic, ArgmaxCubicChartValue>
{};

struct  ArgmaxCubicChart
{
    const unsigned _size;
    const unsigned _n_labels;
    ArgmaxCubicChartValue* _data_top;
    ArgmaxCubicChartValue* _data_bottom;

    ArgmaxCubicChart(const unsigned size, const unsigned n_labels);
    ~ArgmaxCubicChart();

    // top
    inline unsigned offset(unsigned i, unsigned j) const noexcept;
    inline ArgmaxCubicChartValue& operator()(const unsigned i, const unsigned j) noexcept;

    // bottom
    inline unsigned offset(unsigned i, unsigned j, unsigned label) const noexcept;
    inline ArgmaxCubicChartValue& operator()(const unsigned i, unsigned k, unsigned label) noexcept;
};

#include "chart-impl.h"