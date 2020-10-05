
#include "disc_pstruct/data.h"
#include "disc_pstruct/corpus.h"
#include "disc_pstruct/reconstruct.h"
#include "disc_pstruct/set.h"
#include "disc_pstruct/argmax-disc/algorithms.h"
#include "disc_pstruct/inside-outside-disc/algorithms.h"
#include "disc_pstruct/binarize.h"

#include "utils/torch-indexing.h"

PyObject* read_data(PyObject* py_path, const bool fix_parenthesis)
{
    std::string path(PyUnicode_AsUTF8(py_path));
    auto data = read_file(path, fix_parenthesis);

    for (auto& sentence : data)
        sentence.tree.merge_unaries();

    PyObject *py_data = PyList_New(data.size());
    for (unsigned index = 0 ; index < data.size() ; ++ index)
    {
        const auto& sentence = data.at(index);
        // extract words and tags
        PyObject* py_words = PyList_New(sentence.size());
        PyObject* py_tags = PyList_New(sentence.size());
        for (unsigned w_index = 0 ; w_index < sentence.size() ; ++w_index)
        {
            PyObject* word = PyUnicode_FromString(sentence.at(w_index).word.c_str());
            PyObject* tag = PyUnicode_FromString(sentence.at(w_index).postag.c_str());
            // Steal reference so no need to call decref
            PyList_SetItem(py_words, w_index, word);
            PyList_SetItem(py_tags, w_index, tag);
        }

        // extract constituents
        const auto p = map_from_sentence(sentence);
        const auto& constituents = p.first;
        PyObject* py_constituents = PyList_New(constituents.size());
        unsigned constituent_counter = 0u;
        for(const auto& p : constituents)
        {
            const int i = std::get<0>(p.first);
            const int j = std::get<1>(p.first);
            const int k = std::get<2>(p.first);
            const int l = std::get<3>(p.first);
            const std::string label = p.second;

            PyObject *py_constituent = PyTuple_New(5);
            // steal ref
            PyTuple_SET_ITEM(py_constituent, 0, PyUnicode_FromString(label.c_str()));
            PyTuple_SET_ITEM(py_constituent, 1, PyLong_FromLong(i));
            PyTuple_SET_ITEM(py_constituent, 2, PyLong_FromLong(j));
            PyTuple_SET_ITEM(py_constituent, 3, PyLong_FromLong(k));
            PyTuple_SET_ITEM(py_constituent, 4, PyLong_FromLong(l));

            // steal ref
            PyList_SetItem(py_constituents, constituent_counter, py_constituent);
            ++ constituent_counter;
        }

        // sentence id
        PyObject* py_id = PyUnicode_FromString(sentence.id.c_str());

        PyObject* py_k_words = PyUnicode_FromString("words");
        PyObject* py_k_tags = PyUnicode_FromString("tags");
        PyObject* py_k_constituents = PyUnicode_FromString("constituents");
        PyObject* py_k_id = PyUnicode_FromString("id");

        PyObject* dict = PyDict_New();
        std::string k_words("words");
        PyDict_SetItem(dict, py_k_words, py_words);
        PyDict_SetItem(dict, py_k_tags, py_tags);
        PyDict_SetItem(dict, py_k_constituents, py_constituents);
        PyDict_SetItem(dict, py_k_id, py_id);

        Py_DECREF(py_words);
        Py_DECREF(py_tags);
        Py_DECREF(py_constituents);
        Py_DECREF(py_id);

        Py_DECREF(py_k_words);
        Py_DECREF(py_k_tags);
        Py_DECREF(py_k_constituents);
        Py_DECREF(py_k_id);

        // steal ref
        PyList_SetItem(py_data, index, dict);
    }

    return py_data;
}

void write_data(PyObject* py_list, bool fix_parenthesis)
{
    int n_sentences = PyList_Size(py_list);
    std::vector<Sentence*> sentences(n_sentences);

    for (int i = 0 ; i < n_sentences ; ++i)
        sentences.at(i) = (Sentence*) PyLong_AsVoidPtr(PyList_GetItem(py_list, i));

    for (Sentence* sentence : sentences)
    {
        sentence->tree.unmerge_unaries();
        unbinarize(sentence->tree);
    };

    write(std::cout, sentences, fix_parenthesis);

    for (Sentence* sentence : sentences)
        delete sentence;
}
