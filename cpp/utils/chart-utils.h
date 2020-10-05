#pragma once

// lot of build in stuff to make the code more elegant
// field cst is used to stored information about this chart cell that can be quickly retrieved
// during backpointer reconstruction
template<class T, class A>
struct ArgmaxChartValueTemplate
{
    float total_weight;
    float elem_weight;
    T cst;
    int label = -1;
    int rule = -1;
    A* a1;
    A* a2;

    ArgmaxChartValueTemplate();
    ArgmaxChartValueTemplate(const ArgmaxChartValueTemplate<T, A>&) = delete;
    ArgmaxChartValueTemplate(ArgmaxChartValueTemplate<T, A>&&) = delete;

    inline void reset(float w, int l, const bool implicit_binarization = false);
    inline void reset(const std::pair<float, int>& p, const bool implicit_binarization = false);
    inline void reset(float w, const bool implicit_binarization = false);
    inline void axiom(float w, int l);
    inline void axiom(const std::pair<float, int>& p);
    inline void axiom(float w);
    inline void deduce(A& o1, A& o2, const int _rule, const float deduce_weight=0., const bool ignore_elem_weight=false);

    template <class Op>
    void callback(Op op, const bool null_labels = false) const;
    template <class Op>
    void callback_full(Op op, const bool null_labels = false) const;
};

#include "chart-utils-impl.h"