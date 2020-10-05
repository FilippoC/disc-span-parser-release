#include <iostream>
#include <string>
#include <utility>
#include <algorithm>

#include "disc_pstruct/corpus.h"
#include "disc_pstruct/data.h"
#include "disc_pstruct/set.h"
#include "disc_pstruct/argmax-disc/chart.h"
#include "disc_pstruct/argmax-disc/algorithms.h"
#include "cxxtimer.h"

void increment_rules(const ArgmaxDiscChartValue& cell, unsigned* rule_counts)
{
    if (cell.rule >= 0)
        ++rule_counts[cell.rule];
    if (cell.a1 != nullptr)
        increment_rules(*cell.a1, rule_counts);
    if (cell.a2 != nullptr)
        increment_rules(*cell.a2, rule_counts);
}

void increment_sentence_rules_rec(const ArgmaxDiscChartValue& cell, bool* used_rules)
{
    if (cell.rule >= 0)
        used_rules[cell.rule] = true;
    if (cell.a1 != nullptr)
        increment_sentence_rules_rec(*cell.a1, used_rules);
    if (cell.a2 != nullptr)
        increment_sentence_rules_rec(*cell.a2, used_rules);
}

void increment_sentence_rules(const ArgmaxDiscChartValue& cell, unsigned* rule_counts)
{
    bool used_rules[12] = {false};
    increment_sentence_rules_rec(cell, used_rules);
    for (unsigned i = 0 ; i < 12 ; ++i)
        if (used_rules[i])
            ++ rule_counts[i];
}

const char* RULE_NAME[] = {
        "RULE_COMBINE",
        "RULE_CREATE_GAP",
        "RULE_FILL_GAP",
        "RULE_WRAPPING",
        "RULE_COMBINE_KEEP_LEFT",
        "RULE_COMBINE_KEEP_RIGHT",
        "RULE_COMBINE_SHRINK_LEFT",
        "RULE_COMBINE_SHRINK_RIGHT",
        "RULE_IL_NO_GAP",
        "RULE_IL_GAP_CENTER",
        "RULE_IL_GAP_LEFT",
        "RULE_IL_GAP_RIGHT"
};


int main(int argc, char** argv) {
    if (argc != 2)
    {
        std::cerr << "Exactly one parameter required: the path to the treebank file\n";
        return 1;
    }
    std::string path(argv[1]);

    // read data
    auto data = read_file(path, false);

    float n_constituents = 0.;
    float n_bd1_constituents = 0.;
    float n_bd2_constituents = 0.;
    float n_bd3_constituents = 0.;

    unsigned max_sentence_size = 0;

    // count constituents
    std::vector<bool> is_sentence_bd1, is_sentence_bd2, is_sentence_bd3;
    std::vector<float> n_constituents_in_sentence;
    for (auto& sentence : data)
    {
        if (sentence.size() > max_sentence_size)
            max_sentence_size = sentence.size();

        bool sbd1 = true;
        bool sbd2 = true;
        bool sbd3 = true;
        n_constituents_in_sentence.emplace_back(0.);
        sentence.tree.for_each_node([&](Node& node) {
           if (node.label != "ROOT" && node.label != "TOP" && node.label != "NULL")
           {
               ++n_constituents;
               ++ n_constituents_in_sentence.back();

               const auto bd = node._spans.size();
               if (bd == 1)
                   ++ n_bd1_constituents;
               if (bd == 2)
               {
                   sbd1 = false;
                   ++ n_bd2_constituents;
               }
               if (bd == 3)
               {
                   sbd1 = false;
                   sbd2 = false;
                   ++ n_bd3_constituents;
               }
               if (bd > 3)
                   sbd3 = false;
           }
        });
        is_sentence_bd1.push_back(sbd1);
        is_sentence_bd2.push_back(sbd2);
        is_sentence_bd3.push_back(sbd3);

        sentence.tree.merge_unaries();
    }

    std::cout << "path: " << path << "\n";

    std::cout << "----\n";

    std::cout << "n constituents: " << n_constituents << "\n";
    std::cout << "n bd=1 constituents: " << n_bd1_constituents << " (" << 100.f * n_bd1_constituents / n_constituents << "%)\n";
    std::cout << "n bd=2 constituents: " << n_bd2_constituents << " (" << 100.f * n_bd2_constituents / n_constituents << "%)\n";
    std::cout << "n bd=3 constituents: " << n_bd3_constituents << " (" << 100.f * n_bd3_constituents / n_constituents << "%)\n";
    float tmp = n_bd1_constituents + n_bd2_constituents;
    std::cout << "n bd<=2 constituents: " << tmp << " (" << 100.f * tmp / n_constituents << "%)\n";
    tmp = n_bd1_constituents + n_bd2_constituents + n_bd3_constituents;
    std::cout << "n bd<=3 constituents: " << tmp << " (" << 100.f * tmp / n_constituents << "%)\n";
    tmp = n_constituents - n_bd1_constituents - n_bd2_constituents - n_bd3_constituents;
    std::cout << "n bd>3 constituents: " << tmp << " (" << 100.f * tmp / n_constituents << "%)\n";

    std::cout << "----\n";

    std::cout << "n sentences: " << data.size() << "\n";
    tmp = (float) std::count(is_sentence_bd1.begin(), is_sentence_bd1.end(), true);
    std::cout << "n bd=1 sentences: " << tmp << " (" << 100.f * tmp / (float) data.size() << "%)\n";
    tmp = (float) std::count(is_sentence_bd2.begin(), is_sentence_bd2.end(), true);
    std::cout << "n bd<=2 sentences: " << tmp << " (" << 100.f * tmp / (float) data.size() << "%)\n";
    tmp = (float) std::count(is_sentence_bd3.begin(), is_sentence_bd3.end(), true);
    std::cout << "n bd<=3 sentences: " << tmp << " (" << 100.f * tmp / (float) data.size() << "%)\n";
    std::cout << "n bd>3 sentences: " << data.size() - tmp << " (" << 100.f * (data.size() - tmp) / (float) data.size() << "%)\n";

    std::cout << "----\n";

    auto chart = new ArgmaxDiscChart(max_sentence_size);
    for (int complexity = 3 ; complexity <= 6 ; ++complexity)
    {
        for (bool ill_nested : {false, true}) {
            if (complexity == 3 && !ill_nested)
                continue;

            std::cout << "\n-------\n\n";

            std::cout << "Complexity: " << complexity << "\n";
            std::cout << "Ill nested: " << (ill_nested ? "true" : "false") << "\n";

            unsigned rule_counts[12] = {0};
            unsigned rule_sentence_counts[12] = {0};
            float n_parsed_constituents = 0.;
            float n_parsed_trees = 0.;
            float n_parsed_bd2_trees = 0.;
            float n_parsed_bd2_cst = 0.;
            float n_parsed_trees_taking_bd2_only = 0.;

            cxxtimer::Timer timer;
            for (unsigned sentence_index = 0 ; sentence_index < data.size() ; ++sentence_index) {
                const auto &sentence = data.at(sentence_index);
                float n_bd2 = 0.f;
                sentence.tree.for_each_node([&](const Node& node) {
                    if (node.label != "ROOT" && node.label != "TOP" && node.label != "NULL")
                    {
                        if (node._spans.size() <= 2)
                        {
                            float n = 1.f + std::count(node.label.begin(), node.label.end(), '/');
                            n_bd2 += n;
                        }
                    }
                });

                // extract constituents
                const auto p = map_from_sentence(sentence);
                const auto &constituents = p.first;

                timer.start();
                auto &goal = argmax_disc(
                        *chart,
                        sentence.size(),
                        [&](int i, int k, int l, int j) -> std::pair<float, int> {
                            auto it = constituents.find(std::make_tuple(i, k, l, j));
                            if (it != constituents.end()) {
                                float v = 1.f + std::count(it->second.begin(), it->second.end(), '/');
                                return std::make_pair(v, (int) v);
                            }
                            return std::make_pair(-1., (int) 0);
                        },
                        complexity,
                        ill_nested
                );
                timer.stop();

                goal.callback([&](const ChartCstDisc& cst, const int label) {
                    if (std::get<1>(cst) >= 0)
                        n_parsed_bd2_cst += label;
                }, false);

                n_parsed_constituents += goal.total_weight;
                if (goal.total_weight == n_constituents_in_sentence.at(sentence_index))
                {
                    ++n_parsed_trees;
                    if (is_sentence_bd2.at(sentence_index))
                        ++n_parsed_bd2_trees;
                }
                if (goal.total_weight == n_bd2)
                    ++n_parsed_trees_taking_bd2_only;
                increment_rules(goal, rule_counts);
                increment_sentence_rules(goal, rule_sentence_counts);
            }

            std::cout << "Parsed constituents: " << n_parsed_constituents << " (" << 100.f * n_parsed_constituents / n_constituents << "%)\n";
            std::cout << "Parsed bd=2 constituents: " << n_parsed_bd2_cst  << " (" << 100.f * n_parsed_bd2_cst / n_bd2_constituents << "%)\n";
            std::cout << "Parsed bd<=2 constituents: " << n_parsed_constituents << " (" << 100.f * n_parsed_constituents / (n_bd1_constituents + n_bd2_constituents) << "%)\n";

            std::cout << "Parsed tree: " << n_parsed_trees << " (" << 100.f * n_parsed_trees / (float) data.size() << "%)\n";
            float tmp = std::count(is_sentence_bd2.begin(), is_sentence_bd2.end(), true);
            std::cout << "Parsed bd<=2 trees: " << n_parsed_trees << " (" << 100.f * n_parsed_trees / tmp << "%)\n";
            std::cout << "Parsed trees looking only at bd<=2: " << n_parsed_trees_taking_bd2_only << " (" << 100.f * n_parsed_trees_taking_bd2_only / (float) data.size() << "%)\n";
            tmp = tmp - std::count(is_sentence_bd1.begin(), is_sentence_bd1.end(), true);
            std::cout << "Parsed bd=2 trees: " << n_parsed_bd2_trees << " (" << 100.f * n_parsed_bd2_trees / tmp << "%)\n";

            float total = 0.f;
            for (unsigned i = 0; i < 12; ++i)
                total += rule_counts[i];
            for (unsigned i = 0; i < 12; ++i)
            {
                std::cout << RULE_NAME[i] << ":\t";
                std::cout << rule_counts[i] << " (" << 100.f * rule_counts[i] / total << "%)";
                std::cout << "\t-\t";
                std::cout << rule_sentence_counts[i] << " (" << 100.f * rule_sentence_counts[i] / (float) data.size()
                          << "%)";
                std::cout << "\n";
            }
            std::cout << "Parsing time:\t" << timer.count<std::chrono::seconds>() << " seconds\t-\t" << timer.count<std::chrono::minutes>() << " minutes\n";
            std::cout << std::flush;
        }
    }
    return 0;
}