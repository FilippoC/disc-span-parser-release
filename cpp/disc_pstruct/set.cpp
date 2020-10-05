#include "disc_pstruct/set.h"

#include <utility>

std::pair<SpanMap, unsigned> map_from_sentence(const Sentence& sentence, bool disc_only)
{
    SpanMap gold_spans;
    unsigned unreachable_spans = 0u;
    sentence.tree.for_each_node(
            [&] (const Node& node)
            {
                // this can happen if we have merge several roots
                // or have unattached tokens in the sentence
                if (node.label == "NULL")
                    return;

                // can never be predicted by our parser anyway
                if (node._spans.size() > 2)
                {
                    ++unreachable_spans;
                    return;
                }

                if (node._spans.size() == 0)
                    throw std::runtime_error("Empty span");

                if (disc_only && node._spans.size() == 1)
                    return;

                const std::string node_label(node.label);
                if (node._spans.size() == 1)
                {
                    const auto& s = *node._spans.begin();
                    const auto l = s.first;
                    const auto r = s.second;
                    gold_spans.emplace(std::make_pair(
                            std::make_tuple((int) l, -1, -1, (int) r),
                            node_label
                    ));
                }
                else
                {
                    auto s = node._spans.begin();
                    const auto l = s->first;
                    const auto gap_l = s->second;
                    ++s;
                    const auto gap_r  = s->first;
                    const auto r = s->second;
                    gold_spans.emplace(std::make_pair(
                            std::make_tuple((int) l, (int) gap_l, (int) gap_r, (int) r),
                            node_label
                    ));
                }
            }
    );
    return std::make_pair(gold_spans, unreachable_spans);
}

// this  function use the fact that the maps are stored in sorted order for spans
// to accelerate computation
std::pair<SpanMap, SpanMap> difference(const SpanMap& gold_spans, const SpanMap& pred_spans)
{
    SpanMap not_in_gold;
    SpanMap not_in_pred;

    auto it_gold = std::begin(gold_spans);
    auto it_pred = std::begin(pred_spans);
    while (!(it_gold == std::end(gold_spans) && it_pred == std::end(pred_spans)))
    {
        if (it_gold == std::end(gold_spans))
        {
            // this span was predicted but not in gold,
            not_in_gold.emplace(it_pred->first, it_pred->second);
            ++it_pred;
        }
        else if (it_pred == std::end(pred_spans))
        {
            // this span was in gold but not predicted
            not_in_pred.emplace(it_gold->first, it_gold->second);
            ++it_gold;
        }
        else
        {
            if (it_gold->first == it_pred->first)
            {
                if (it_gold->second != it_pred->second)
                {
                    // predicted labels are different
                    not_in_gold.emplace(it_pred->first, it_pred->second);
                    not_in_pred.emplace(it_gold->first, it_gold->second);
                }
                ++ it_gold;
                ++ it_pred;
            }
            else if (it_gold->first < it_pred->first)
            {
                // this gold span was not predicted
                not_in_pred.emplace(it_gold->first, it_gold->second);
                ++it_gold;
            }
            else
            {
                // this span was predicted but not in gold
                not_in_gold.emplace(it_pred->first, it_pred->second);
                ++it_pred;
            }
        }
    }

    return std::make_pair(not_in_gold, not_in_pred);
}
