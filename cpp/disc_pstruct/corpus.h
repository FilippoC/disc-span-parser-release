#pragma once

#include <boost/core/noncopyable.hpp>
#include <vector>
#include <string>
#include "tree.h"

bool is_export_punct(const std::string& word, const std::string& tag);

struct Token
{
    std::string word;
    std::string postag;
    // true if covered by the parser tree
    // e.g. we ignore punctuation
    bool covered;

    Token(const std::string& word, const std::string& postag, const bool covered) :
            word(word),
            postag(postag),
            covered(covered)
    {}
};


struct Sentence :
        public std::vector<Token>,
        private boost::noncopyable
{
    const std::string id;
    Tree tree;

    Sentence() = delete;
    Sentence(Sentence&& sentence) :
            std::vector<Token>(sentence),
            id(sentence.id),
            tree(std::move(sentence.tree))
    {}

    template<class ItToken>
    Sentence(
            const std::string& id,
            ItToken begin_token, ItToken end_token,
            std::shared_ptr<Node> root
    ) :
            std::vector<Token>(begin_token, end_token),
            id(id),
            tree(size(), root)
    {}

    template<class ItToken, class ItRel, class ItPhrase>
    Sentence(
            std::string& id,
            ItToken begin_token, ItToken end_token,
            ItRel begin_rel, ItRel end_rel,
            ItPhrase begin_phrase, ItPhrase end_phrase
    ) :
            std::vector<Token>(begin_token, end_token),
            id(id),
            tree(begin_rel, end_rel, begin_phrase, end_phrase)
    {}

};

void read_file(const std::string&, std::vector<Sentence>& output, bool fix_parenthesis=false, bool uncover_punct=false);
std::vector<Sentence> read_file(const std::string& path, bool fix_parenthesis=false, bool uncover_punct=false);
void write(std::ostream& os, std::vector<Sentence*>& output, bool fix_parenthesis=false);
