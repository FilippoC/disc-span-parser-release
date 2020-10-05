#include "tree.h"
#include <boost/algorithm/string.hpp>
#include <iostream>

Rel::Rel(std::string name, unsigned parent_id) :
    name(name),
    parent_id(parent_id)
{}
Phrase::Phrase(unsigned id, std::string tag) :
    id(id),
    tag(tag)
{}
Span::Span(unsigned i, unsigned j) :
    std::pair<unsigned, unsigned>(i, j)
{
    if (j < i)
        throw std::runtime_error("Invalid span");
}
Span::Span(unsigned i) :
    std::pair<unsigned, unsigned>(i, i)
{}

std::ostream& operator<<(std::ostream& os, const Span& span)  
{  
    os << "(" << span.first << "," << span.second << ")";
    return os;  
} 
bool Spans::has_overlap()
{
    if (size() <= 1u)
        return false;

    auto it = begin();
    unsigned right = it->second;
    ++it;
    for (; it != end() ; ++it)
    {
        if (it->first <= right)
            return true;
        else
            right = it->second;
    }

    return false;
}

void Spans::merge_contiguous()
{
    if (size() <= 1u)
        return;

    Spans o;
    auto it = begin();
    Span previous = *it;
    ++it;
    for (; it != end() ; ++it)
    {
        Span current = *it;
        // works even if overlapping
        if (current.first <= previous.second + 1)
        {
            previous.second = current.second;
        }
        else
        {
            o.insert(previous);
            previous = current;
        }
    }
    o.insert(previous);
    swap(o);
}
bool Spans::operator==(const Spans& o)
{
    if (size() != o.size())
        return false;
    auto it = begin();
    auto it2 = o.begin();
    while (it != end())
    {
        if (*it != *it2)
            return false;
        ++it;
        ++it2;
    }
    return true;
}

std::ostream& operator<<(std::ostream& os, const Spans& spans)
{  
    for(const auto& s : spans)
        os << s;
    return os;  
} 

Node::Node(unsigned i)
{
    _spans.emplace(i, i);
}

Node::Node()
{}


void Node::add_child(std::shared_ptr<Node> c)
{
    if (c->_parent != nullptr)
        throw std::runtime_error("Node already has a parent");

    _children.emplace_back(c);
    c->_parent = this;
}

void Node::remove_empty(bool recursive)
{
    if (is_leaf())
        return;
    else
    {
        //_spans.clear();
        std::vector<std::shared_ptr<Node>> backup;
        _children.swap(backup);
        for (auto& c : backup)
        {
            if (c->_spans.size() == 0)
                continue;
            _children.push_back(c);
            if (recursive)
                c->remove_empty();
        }
    }
}


void Node::compute_spans(bool recursive)
{
    if (is_leaf())
    {
        //if (_spans.size() == 0)
        //    throw std::runtime_error("Empty span for leaf node");
        _spans.merge_contiguous();
    }
    else
    {
        //_spans.clear();
        for (auto& c : _children)
        {
            if (recursive)
                c->compute_spans();
            _spans.insert(c->_spans.begin(), c->_spans.end());
        }
        if (_spans.has_overlap())
            throw std::runtime_error("Overlapping spans");
        _spans.merge_contiguous();
    }
}

bool Node::is_leaf() const
{
    return _children.size() == 0u;
}

unsigned Node::block_degree() const
{
    unsigned ret = _spans.size();
    for (auto& c : _children)
        ret = std::max(ret, c->block_degree());
    return ret;
}

void Node::merge_unaries()
{
    if (is_leaf())
        return;
    for (auto& c : _children)
        c->merge_unaries();
    if (_children.size() == 1u && _children.at(0)->_spans == _spans)
    {
        std::shared_ptr<Node> o = _children.at(0u);
        _children = o->_children;
        label = label + "/" + o->label;
        _spans = o->_spans;

        for (auto& c : _children)
            c->_parent = this;
    }
}

void Node::unmerge_unaries()
{
    if (is_leaf())
        return;
    for (auto& c : _children)
        c->unmerge_unaries();
    if (label.find("/") != std::string::npos)
    {
        std::vector<std::string> labels;
        boost::split(labels,label,boost::is_any_of("/"));

        std::vector<std::shared_ptr<Node>> children_backup;
        children_backup.swap(_children);
        label = labels.at(0);
        Node* last = this;
        for (unsigned i = 1; i < labels.size() ; ++ i)
        {
            std::shared_ptr<Node> n = std::make_shared<Node>();
            n->_spans = last->_spans;
            n->label = labels.at(i);
            last->add_child(n);
            last = n.get();
        }
        for (auto c : children_backup)
            last->add_child(c);
    }
}


bool Node::is_binary()
{
    if (_children.size() == 0u)
        return true;
    else if (_children.size() >= 3u)
        return false;
    else if (std::all_of(
            _children.begin(), _children.end(),
            [] (std::shared_ptr<Node> c)
            {
                return c->is_binary();
            }
    ))
        return true;
    else
        return false;

}

bool Node::no_gap_propagation()
{
    if (is_leaf())
        return true;

    if (!std::all_of(
        _children.begin(), _children.end(),
        [] (std::shared_ptr<Node> n)
        {
            return n->no_gap_propagation();
        }
    ))
        return false;

    if (_spans.size() == 1u)
        return true;
    else if (_parent != nullptr && _parent->_spans.size() == 1)
        return true;
    else
        return false;
}

std::ostream& operator<<(std::ostream& os, const Node& node)  
{  
    os << node.label << "=" << node._spans << std::endl;
    if (!node.is_leaf())
    {
        os << "--CHILDREN--" << std::endl;
        for (const auto& c : node._children)
            os << *c;
        os << "--END CHILDREN--" << std::endl;
    }
    return os;  
} 

Tree::Tree(Tree&& tree) :
    _size(tree._size),
    _roots(std::move(tree._roots))
{}

void Tree::merge_unaries()
{
    for (auto& n : _roots)
        n->merge_unaries();
}

void Tree::unmerge_unaries()
{
    for (auto& n : _roots)
        n->unmerge_unaries();
}

unsigned Tree::block_degree()
{
    auto it = _roots.begin();
    unsigned ret = (*it)->block_degree();
    ++it;
    for(; it != _roots.end() ; ++it)
    {
        ret = std::max(ret, (*it)->block_degree());
    }
    return ret;
}

bool Tree::is_binary()
{
    return std::all_of(
        _roots.begin(), _roots.end(),
        [] (std::shared_ptr<Node> n)
        {
            return n->is_binary();
        }
    );
}
bool Tree::no_gap_propagation()
{
    return std::all_of(
        _roots.begin(), _roots.end(),
        [](std::shared_ptr<Node> n)
        {
            return n->no_gap_propagation();
        }
    );
}
