#pragma once

#include <map>
#include <tuple>
#include "disc_pstruct/corpus.h"

// build a set with gold spans
typedef  std::map<std::tuple<int, int, int, int>, std::string> SpanMap;
typedef  std::map<std::tuple<int, int, int, int>, int> SpanMapInt;

std::pair<SpanMap, unsigned> map_from_sentence(const Sentence& sentence, bool disc_only = false);