#include <iostream>
#include "corpus.h"


namespace
{
const std::vector<std::string> negra_tags= {
        "$,", "$(", "$[", "$."
};

const std::vector<std::string> alpino_tags = {
        "PUNCT", "punct", "LET[]", "LET()", "LET",
        "let[]", "let()", "let"
};

const std::vector<std::string> ptb_tags = {
        ",", ":", "``", "''", ".", "-NONE-"
};

const std::vector<std::string> words = {
        ".", ",", ":", ";", "'", "`", "\"", "``",
        "''", "-", "(", ")", "/", "&", "$", "!",
        "!!!", "!!!", "?", "??", "???", "..",
        "...", "«", "»"
};
}

bool is_export_punct(const std::string& word, const std::string& tag)
{
    return
            std::any_of(
                    std::begin(negra_tags), std::end(negra_tags),
                    [&] (const std::string& v) { return tag == v; }
            )
            ||
            std::any_of(
                    std::begin(alpino_tags), std::end(alpino_tags),
                    [&] (const std::string& v) { return tag == v; }
            )
            ||
            std::any_of(
                    std::begin(ptb_tags), std::end(ptb_tags),
                    [&] (const std::string& v) { return tag== v; }
            )
            ||
            std::any_of(
                    std::begin(words), std::end(words),
                    [&] (const std::string& v) { return word == v; }
            )
            ;
}


std::vector<Sentence> read_file(const std::string& path, bool fix_parenthesis, bool uncover_punct)
{
    std::vector<Sentence> output;
    read_file(path, output, fix_parenthesis, uncover_punct);
    return output;
}

void read_file(const std::string& path, std::vector<Sentence>& output, bool fix_parenthesis, bool uncover_punct)
{
    std::string line;
    std::ifstream is(path);
    if (!is.is_open())
        throw std::runtime_error("Could not open file: " + path);

    while (std::getline(is, line)) {
        //std::cout << "R: " << line << std::endl;
        // ignore empty line
        if (line.length() <= 0)
            continue;

        // ignore comments
        if (boost::starts_with(line, "%%"))
            continue;

        // ignore format
        if (boost::starts_with(line, "#FORMAT"))
            continue;

        // beginning of table
        if (boost::starts_with(line, "#BOT"))
        {
            bool in_table = true;
            while (std::getline(is, line))
            {
                if (boost::starts_with(line, "#EOT"))
                {
                    in_table = false;
                    break;
                }
            }
            if (in_table)
                throw std::runtime_error("EOF while in table");
            continue;
        }

        // beginning of sentence
        if (boost::starts_with(line, "#BOS"))
        {
            bool in_sentence = true;

            // format: #BOS num editor date origin
            // we only care about the num
            std::string sent_id;
            std::istringstream(line) >> sent_id >> sent_id;

            std::vector<Token> tokens;
            std::vector<Rel> token_rels;
            std::vector<std::pair<Phrase, Rel>> phrases;
            try
            {
                while (std::getline(is, line))
                {
                    if (boost::starts_with(line, "#EOS"))
                    {
                        in_sentence = false;
                        break;
                    }

                    if (boost::starts_with(line, "#") && !std::isspace(line.at(1)))
                    {
                        // phrase
                        // #num tag morphtag reltype parent
                        std::stringstream ss(line);
                        ss.ignore(); // ignore the starting #

                        unsigned num;
                        std::string unused;
                        std::string tag;
                        std::string morph;
                        std::string reltype;
                        unsigned parent;
                        ss >> num >> unused >> tag >> morph >> reltype >> parent;

                        phrases.emplace_back(std::make_pair<Phrase, Rel>(
                                {num, tag}, {reltype, parent}
                        ));
                    }
                    else
                    {
                        // word
                        std::stringstream ss(line);
                        // word tag morph edge parent
                        std::string word;
                        std::string lemma;
                        std::string tag;
                        std::string morph;
                        std::string reltype;
                        unsigned parent;
                        // REMOVE the second tag
                        ss >> word >> lemma >> tag >> morph >> reltype >> parent;

                        // replace parenthesis if needed
                        if (fix_parenthesis)
                        {
                            if (word == "-LRB-" or word == "#LRB#")
                                word = "(";
                            else if (word == "-RRB-" or word == "#RRB#")
                                word = ")";
                        }

                        bool covered = true;
                        if (uncover_punct)
                            covered = !is_export_punct(word, tag);
                        // WARNING: what is this check for?
                        //if (parent == 0u && covered)
                        //    throw std::runtime_error("Unattached token that is not punct");
                        tokens.emplace_back(word, tag, covered);
                        //token_rels.emplace_back(reltype, (covered ? parent : 0u));
                        if (covered)
                            token_rels.emplace_back(reltype, parent);
                    }
                }
            }
            catch (const std::runtime_error& e)
            {
                std::cerr << "CURRENT ID: " << sent_id << std::endl;
                throw e;
            }
            if (in_sentence)
                throw std::runtime_error("EOF while in sentence");
            try
            {
                // we only add a sentence if it has covered tokens
                if (token_rels.size() > 0)
                {
                    output.emplace_back(
                            sent_id,
                            tokens.begin(), tokens.end(),
                            token_rels.begin(), token_rels.end(),
                            phrases.begin(), phrases.end()
                    );
                }
            }
            catch (const std::runtime_error& e)
            {
                std::cerr << "CURRENT ID: " << sent_id << std::endl;
                throw e;
            }
            continue;
        }

        throw std::runtime_error("Error in data");
    }

    is.close();
}

namespace
{
void write_node(
        std::ostream& os,
        std::shared_ptr<Node> node,
        unsigned head,
        unsigned* next,
        std::vector<unsigned>* heads,
        bool output
)
{
    unsigned id = *next;
    if (output)
        os
                << "#"
                << id
                << "\t"
                << "--"
                << "\t"
                << node->label
                << "\t"
                << "--"
                << "\t"
                << "--"
                << "\t"
                << head
                << "\n"
                ;
    *next += 1;

    // we visit in top-down order, so it will be rewritten if needed
    for (auto const& s : node->_spans)
        for (unsigned i = s.first ; i <= s.second ; ++ i)
            heads->at(i) = id;
    if (!node->is_leaf())
        for (std::shared_ptr<Node> c : node->_children)
            write_node(os, c, id, next, heads, output);
}
}

void write(std::ostream& os, std::vector<Sentence*>& output, bool fix_parenthesis)
{
    for (Sentence* ptr_sentence : output)
    {
        const Sentence& sentence = *ptr_sentence;
        os << "#BOS " << sentence.id << "\n";

        std::vector<unsigned> heads(sentence.size(), 999);

        // structure
        unsigned next = 500u;
        for (std::shared_ptr<Node> node : sentence.tree._roots)
            write_node(os, node, 0, &next, &heads, false);

        // tokens
        unsigned j = 0u;
        for (unsigned i = 0u ; i < sentence.size() ; ++i)
        {
            const auto& token = sentence[i];
            unsigned head = (token.covered ? heads.at(j) : 0u);

            // if there is no structure at all over the sentence
            if (head == 999)
                head = 0u;

            std::string word = token.word;

            // replace parenthesis if needed
            if (fix_parenthesis)
            {
                if (word == "(")
                    word = "-LRB-";
                else if (word == ")")
                    word = "-RRB-";
            }

            os
                    << word
                    << "\t"
                    << "--" // lemme
                    << "\t"
                    << token.postag
                    << "\t"
                    << "--"
                    << "\t"
                    << "--"
                    << "\t"
                    << head
                    << "\n"
                    ;
            if (token.covered)
                ++ j;
        }


        // structure
        next = 500u;
        for (std::shared_ptr<Node> node : sentence.tree._roots)
            write_node(os, node, 0, &next, &heads, true);

        os << "#EOS " << sentence.id << "\n";
    }

}
