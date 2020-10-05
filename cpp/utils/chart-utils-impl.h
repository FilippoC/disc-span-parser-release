#pragma once

#include <iostream>

template<class T, class A>
ArgmaxChartValueTemplate<T, A>::ArgmaxChartValueTemplate() :
        total_weight(-std::numeric_limits<float>::infinity()),
        elem_weight(0.f),
        a1(nullptr),
        a2(nullptr)
{}

template<class T, class A>
void ArgmaxChartValueTemplate<T, A>::reset(float w, int l, const bool implicit_binarization)
{
    if (not implicit_binarization)
    {
        elem_weight = w;
        label = (w >= 0. ? l : -1);
    }
    else if (w >= 0.)
    {
        elem_weight = w;
        label = l;
    }
    else
    {
        elem_weight = 0;
        label = -1;
    }
    total_weight = -std::numeric_limits<float>::infinity();
    a1 = nullptr;
    a2 = nullptr;
}

template<class T, class A>
inline void ArgmaxChartValueTemplate<T, A>::reset(const std::pair<float, int>& p, const bool implicit_binarization)
{
    reset(p.first, p.second, implicit_binarization);
}

template<class T, class A>
void ArgmaxChartValueTemplate<T, A>::reset(float w, const bool implicit_binarization)
{
    reset(w, 0, implicit_binarization);
}

template<class T, class A>
void ArgmaxChartValueTemplate<T, A>::axiom(float w, int l)
{
    if (w > 0)
    {
        elem_weight = w;
        total_weight = w;
        label = l;
    }
    else
    {
        elem_weight = 0;
        total_weight = 0;
        label = -1;
    }
}

template<class T, class A>
inline void ArgmaxChartValueTemplate<T, A>::axiom(const std::pair<float, int>& p)
{
    axiom(p.first, p.second);
}

template<class T, class A>
void ArgmaxChartValueTemplate<T, A>::axiom(float w)
{
    axiom(w, 0);
}

template<class T, class A>
void ArgmaxChartValueTemplate<T, A>::deduce(A& o1, A& o2, const int _rule, const float deduction_weight, const bool ignore_elem_weight)
{
    const float s = o1.total_weight + o2.total_weight + deduction_weight + (ignore_elem_weight ? 0. : elem_weight);
    if (s > total_weight)
    {
        //if (deduction_weight)
        //    throw std::runtime_error("YES!");
        total_weight = s;
        a1 = &o1;
        a2 = &o2;
        rule = _rule;
    }
}

template <class T, class A>
template <class Op>
void ArgmaxChartValueTemplate<T, A>::callback(Op op, const bool null_labels) const
{
    // we consider this valid only if the label is not null
    if (label >= 0 || null_labels)
        op(cst, label);
    if (a1 != nullptr)
        a1->callback(op, null_labels);
    if (a2 != nullptr)
        a2->callback(op, null_labels);
}

template <class T, class A>
template <class Op>
void ArgmaxChartValueTemplate<T, A>::callback_full(Op op, const bool null_labels) const
{
    // we consider this valid only if the label is not null
    if (label >= 0 || null_labels)
        op(*static_cast<const A*>(this));
    if (a1 != nullptr)
        a1->callback_full(op, null_labels);
    if (a2 != nullptr)
        a2->callback_full(op, null_labels);
}