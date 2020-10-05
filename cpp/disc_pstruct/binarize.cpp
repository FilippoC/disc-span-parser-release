#include "disc_pstruct/binarize.h"

// empty node at the right
void binarize(std::shared_ptr<Node> node, bool right)
{
    if (node->is_leaf())
        return;

    for (auto c : node->_children)
        binarize(c, right);

    if (node->_children.size() <= 2)
        return;

    auto empty = std::make_shared<Node>();

    if (right)
    {
        // skip the first one
        std::sort(
            node->_children.begin(),
            node->_children.end(),
            [] (const std::shared_ptr<Node> a, const std::shared_ptr<Node> b)
            {
                return a->_spans.begin()->first < b->_spans.begin()->first;
            }
        );
        auto it_begin = node->_children.begin();
        ++it_begin;

        std::copy(
            it_begin, node->_children.end(),
            std::back_inserter(empty->_children)
        );
    }
    else
    {
        // skip the last one
        std::sort(
            node->_children.begin(),
            node->_children.end(),
            [] (const std::shared_ptr<Node> a, const std::shared_ptr<Node> b)
            {
                // reverse sort
                return b->_spans.begin()->first < a->_spans.begin()->first;
            }
        );
        auto it_begin = node->_children.begin();
        ++it_begin;

        std::copy(
            it_begin, node->_children.end(),
            std::back_inserter(empty->_children)
        );
    }
    empty->_parent = node.get();
    empty->_is_root = false;
    empty->label = "NULL";

    node->_children.resize(2);
    node->_children.at(1) = empty;

    // update spans
    empty->compute_spans(false);
    node->compute_spans(false);
}

void binarize_right(std::shared_ptr<Node> node)
{
    binarize(node, true);
}

void binarize_left(std::shared_ptr<Node> node)
{
    binarize(node, false);
}

void binarize(Tree& tree, bool right)
{
    for (auto n : tree._roots)
        binarize(n, right);
}

void binarize_right(Tree& tree)
{
    binarize(tree, true);
}

void binarize_left(Tree& tree)
{
    binarize(tree, false);
}

void unbinarize(Tree& tree)
{
    std::vector<std::shared_ptr<Node>> backup;

    for (const auto& n : tree._roots)
    {
        unbinarize(n);

        if (n->label == "NULL")
        {
            std::copy(
                    n->_children.begin(), n->_children.end(),
                    std::back_inserter(backup)
            );
        }
        else
        {
            backup.push_back(n);
        }
    }

    tree._roots.swap(backup);
}

void unbinarize(std::shared_ptr<Node> node)
{
    if (node->is_leaf())
        return;

    for (auto c : node->_children)
        unbinarize(c);

    std::vector<std::shared_ptr<Node>> backup;
    backup.swap(node->_children);
    for (std::shared_ptr<Node> child : backup)
    {
        if (child->label == "NULL")
        {
            if (child->is_leaf())
            {
                // TODO: why is this commented?
                //node->_spans.insert(child->_spans.begin(), child->_spans.end());
                //node->_spans.merge_contiguous();
            }
            else
            {
                std::copy(
                    child->_children.begin(), child->_children.end(),
                    std::back_inserter(node->_children)
                );
            }
        }
        else
        {
            node->_children.push_back(child);
        }
    }
}
