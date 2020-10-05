#pragma once

#include "disc_pstruct/argmax-disc/chart.h"
#include "disc_pstruct/tree.h"


template<class Op>
std::shared_ptr<Node> reconstruct(const ArgmaxDiscChartValue& cell, Op callback)
{
    std::shared_ptr<Node> node = std::make_shared<Node>();
    node->label = callback(cell.cst,  cell.label);

    if (cell.a1 != nullptr)
    {
        auto c = reconstruct(*(cell.a1), callback);
        node->_children.push_back(c);
    }
    if (cell.a2 != nullptr)
    {
        auto c = reconstruct(*(cell.a2), callback);
        node->_children.push_back(c);
    }

    if (cell.a1 == nullptr && cell.a2 == nullptr)
    {
        auto const& cst = cell.cst;

        if (std::get<1>(cst) < 0)
        {
            node->_spans.emplace(std::get<0>(cst), std::get<3>(cst));
        }
        // what does this condition do? why is it useful?
        // in which case can we have a discontinuous constituent at the bottom of the recursion?
        else
        {
            node->_spans.emplace(std::get<0>(cst), std::get<1>(cst));
            node->_spans.emplace(std::get<2>(cst), std::get<3>(cst));
        }
    }

    return node;
}



template<class Op>
std::shared_ptr<Node> reconstruct_cubic(const ArgmaxCubicChartValue& cell, Op callback)
{
    std::shared_ptr<Node> node = std::make_shared<Node>();
    node->label = callback(cell.label, true);

    if (cell.a1 == nullptr && cell.a2 == nullptr)
    {
        // leaf
        auto const &cst = cell.cst;
        node->_spans.emplace(std::get<0>(cst), std::get<1>(cst));

        return node;
    }
    else if (std::get<2>(cell.a1->cst) == -1)
    {
        // "standard deduction item
        auto c1 = reconstruct_cubic(*(cell.a1), callback);
        node->_children.push_back(c1);

        auto c2 = reconstruct_cubic(*(cell.a2), callback);
            node->_children.push_back(c2);

        return node;
    }
    else
    {
        // bottom item

        // construct discontinuous child
        std::shared_ptr<Node> disc_node = std::make_shared<Node>();
        disc_node->label = callback(std::get<2>(cell.a1->cst), false);

        // retrieve child of the discontinuous node
        auto disc_c1 = reconstruct_cubic(*(cell.a1->a1), callback);
        disc_node->_children.push_back(disc_c1);
        auto disc_c2 = reconstruct_cubic(*(cell.a2), callback);
        disc_node->_children.push_back(disc_c2);

        node->_children.push_back(disc_node);

        // retrieve child in the gap
        auto c1 = reconstruct_cubic(*(cell.a1->a2), callback);
        node->_children.push_back(c1);

        return node;
    }
}