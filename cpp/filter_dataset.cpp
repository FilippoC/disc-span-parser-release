#include <iostream>
#include <string>
#include <utility>
#include <algorithm>

#include "disc_pstruct/corpus.h"
#include "disc_pstruct/data.h"
#include "disc_pstruct/set.h"
#include "disc_pstruct/argmax-disc/chart.h"
#include "disc_pstruct/argmax-disc/algorithms.h"
#include "disc_pstruct/reconstruct.h"
#include "disc_pstruct/binarize.h"
#include "cxxtimer.h"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid number of parameters\n";
        return 1;
    }
    std::string path(argv[1]);
    int complexity = atoi(argv[2]);
    bool ill_nested = false;
    if (std::string("true") == argv[3])
        ill_nested = true;
    else if (std::string("false") != argv[3])
        throw std::runtime_error("Invalid argument");

    // read data
    auto data = read_file(path, false);
    std::vector<Sentence*> new_data;

    unsigned max_sentence_size = 0;
    for (auto& sentence : data)
    {
        if (sentence.size() > max_sentence_size)
            max_sentence_size = sentence.size();
        sentence.tree.merge_unaries();
    }

    auto chart = new ArgmaxDiscChart(max_sentence_size);
    for (unsigned sentence_index = 0 ; sentence_index < data.size() ; ++sentence_index)
    {
        const auto& sentence = data.at(sentence_index);

        // extract constituents
        const auto p = map_from_sentence(sentence);
        const auto &constituents = p.first;

        const ArgmaxDiscChartValue* goal;
        goal = &argmax_disc(
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

        // reconstruct the tree
        const auto root = reconstruct(*goal,
                [&] (const ChartCstDisc& cst, const int label_id) -> std::string
                          {
                              if (label_id < 0)
                                  return "NULL";
                              else
                              {
                                  return constituents.find(cst)->second;
                              }
                          }
        );

        new_data.emplace_back(new Sentence(sentence.id, sentence.begin(), sentence.end(), root));
        new_data.back()->tree.unmerge_unaries();
        unbinarize(new_data.back()->tree);
    }

    write(std::cout, new_data, false);
}