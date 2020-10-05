#pragma once

#include <boost/core/noncopyable.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <vector>
#include <iostream>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <memory>

struct Rel
{
    std::string name;
    unsigned parent_id;

    Rel(std::string name, unsigned parent_id);
};

struct Phrase
{
    unsigned id;
    std::string tag;

    Phrase(unsigned id, std::string tag);
};

struct Span :
    public std::pair<unsigned, unsigned>
{
    Span(unsigned i, unsigned j);
    Span(unsigned i);
};

std::ostream& operator<<(std::ostream& os, const Span& span)  ;

struct Spans :
    public std::set<Span>
{
    bool has_overlap();
    void merge_contiguous();
    bool operator==(const Spans& o);
};

std::ostream& operator<<(std::ostream& os, const Spans& spans);

struct Node :
    private boost::noncopyable
{
    std::vector<std::shared_ptr<Node>> _children;
    std::string label;
    Spans _spans;
    Node* _parent = nullptr;
    bool _is_root = true;

    Node(unsigned i);
    Node();

    void add_child(std::shared_ptr<Node> c);
    void compute_spans(bool recursive=true);
    void remove_empty(bool recursive=true);
    bool is_leaf() const;
    unsigned block_degree() const;
    void merge_unaries();
    void unmerge_unaries();
    bool is_binary();
    bool no_gap_propagation();

    template<class Op>
    void for_each_node(Op op) const
    {
        const auto& o = *this;
        op(o);
        if (!is_leaf())
            for (const auto n : _children)
                n->for_each_node(op);
    }

    template<class Op>
    void for_each_node(Op op)
    {
        auto& o = *this;
        op(o);
        if (!is_leaf())
            for (const auto n : _children)
                n->for_each_node(op);
    }
};

std::ostream& operator<<(std::ostream& os, const Node& node);

struct Tree :
    private boost::noncopyable
{
    unsigned _size;
    std::vector<std::shared_ptr<Node>> _roots;

    Tree(Tree&& tree);

    Tree(unsigned size, std::shared_ptr<Node> root) :
        _size(size)
    {
        root->compute_spans();
        _roots.push_back(root);
    }

    template<class ItToken, class ItPhrase>
    Tree(ItToken begin_token, ItToken end_token, ItPhrase begin_phrase, ItPhrase end_phrase)
    {
        std::map<unsigned, std::shared_ptr<Node>> id_to_node;
        _size = 0u;
        std::vector<unsigned> unattached_tokens;
        for_each(
            begin_token, end_token,
            [&] (const Rel& rel)
            {
                //std::cerr << "T: " << token.word << "\t" << token.pos << "\t" << rel.parent_id << std::endl;

                // REMOVED: just do not give uncovered tokens
                // we ignore tokens that do not appear in the sentence
                //if (rel.parent_id == 0)
                //    return;
                if (rel.parent_id == 0)
                {
                    unattached_tokens.push_back(_size);
                }
                else
                {
                    //std::shared_ptr<Node> node(new Node(_size));
                    auto iter = id_to_node.find(rel.parent_id);
                    if (iter == std::end(id_to_node))
                    {
                        // _size -> add the current position in the span
                        //std::cerr << "C:" << _size << "\t" << rel.parent_id << std::endl;
                        std::shared_ptr<Node> parent(new Node(_size));
                        //parent->add_child(node);
                        id_to_node.emplace(rel.parent_id, parent);
                    }
                    else
                    {
                        //iter->second->add_child(node);
                        // add the current position to the span
                        //std::cerr << "E:" << _size << "\t" << rel.parent_id << std::endl;
                        iter->second->_spans.emplace(_size, _size);
                    }
                }
                ++ _size;
            }
        );
        
        for_each(
            begin_phrase, end_phrase,
            [&] (const std::pair<Phrase, Rel>& p)
            {
                const auto& phrase = p.first;
                const auto& rel = p.second;
                //std::cerr << "T: " << phrase.id << "\t" << phrase.tag << "\t" << rel.parent_id << std::endl;
                if (rel.parent_id == 0u)
                {
                    auto iter = id_to_node.find(phrase.id);
                    if (iter == std::end(id_to_node))
                    {
                        //std::cerr << "E: " << phrase.id << "\t" << rel.parent_id << std::endl;
                        //throw std::runtime_error("root without lexical anchor");
                        auto n = std::make_shared<Node>();
                        n->label = phrase.tag;
                        id_to_node.emplace(phrase.id, n);
                        _roots.emplace_back(n);
                    }
                    else
                    {
                        _roots.emplace_back(iter->second);
                        iter->second->label = phrase.tag;
                    }
                }
                else
                {
                    auto iter_parent = id_to_node.find(rel.parent_id);
                    auto iter_child = id_to_node.find(phrase.id);

                    std::shared_ptr<Node> parent;
                    if (iter_parent == std::end(id_to_node))
                    {
                        parent = std::make_shared<Node>();
                        id_to_node.emplace(rel.parent_id, parent);
                    }
                    else
                        parent = iter_parent->second;

                    std::shared_ptr<Node> child;
                    if (iter_child == std::end(id_to_node))
                    {
                        child = std::make_shared<Node>();
                        child->label = phrase.tag;
                        id_to_node.emplace(phrase.id, child);
                    }
                    else
                    {
                        child = iter_child->second;
                        child->label = phrase.tag;
                    }

                    parent->add_child(child);
                }
            }
        );

        // several roots & unattached tokens
        if (unattached_tokens.size() >= 1 || _roots.size() >= 2u)
        {
            auto new_root = std::make_shared<Node>();
            new_root->_children.swap(_roots);
            new_root->label = "NULL";
            for (std::shared_ptr<Node> n : new_root->_children)
            {
                n->_parent = new_root.get();
                n->_is_root = false;
            }
            for (const auto i : unattached_tokens)
                new_root->_spans.emplace(i);
            _roots.push_back(new_root);
        }

        if (_roots.size() != 1u)
        {
            std::cerr << "ROOT check size: " << _roots.size() << "\n";
            throw std::runtime_error("This should not happen");
        }

        Spans s;
        for (auto& n : _roots)
        {
            n->compute_spans();
            n->remove_empty();
            s.insert(n->_spans.begin(), n->_spans.end());
        }
        if (s.has_overlap())
            throw std::runtime_error("Overlapping trees in sentence");
        s.merge_contiguous();
        if (s.size() != 1u)
        {
            throw std::runtime_error("Trees do not span the sentence");
        }
        if (s.begin()->first != 0 || s.begin()->second != _size - 1)
        {
            throw std::runtime_error("Trees do not span the sentence");
        }
    }

    void replace(std::shared_ptr<Node> new_root)
    {
        new_root->compute_spans();
        _roots.clear();
        _roots.push_back(new_root);
    }


    template<class Op>
    void for_each_node(Op op) const
    {
        for (auto n : _roots)
            n->for_each_node(op);
    }

    void merge_unaries();
    void unmerge_unaries();
    unsigned block_degree();
    bool is_binary();
    bool no_gap_propagation();
};
