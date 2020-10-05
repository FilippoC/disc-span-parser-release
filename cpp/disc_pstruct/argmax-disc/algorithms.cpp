#include "disc_pstruct/argmax-disc/algorithms.h"

#include "disc_pstruct/corpus.h"
#include "disc_pstruct/tree.h"
#include "disc_pstruct/set.h"
#include "utils/torch-indexing.h"
#include "disc_pstruct/reconstruct.h"


PyObject* max_recall(
        ArgmaxDiscChart* chart,
        const unsigned size,
        SpanMapInt* gold,
        const unsigned complexity,
        const bool ill_nested
)
{
    // inline const ArgmaxDiscChartValue& argmax_disc(ArgmaxDiscChart& chart, const unsigned size, Op Weights, const unsigned complexity=6, const bool ill_nested=true)
    const ArgmaxDiscChartValue* goal;

    goal = &argmax_disc(
            *chart,
            size,
            [&](int i, int k, int l, int j) -> std::pair<float, int> {
                auto it = gold->find(std::make_tuple(i, k, l, j));
                if (it != gold->end())
                {
                    return std::make_pair(1.f, 1);
                }
                return std::make_pair(-1., 0);
            },
            complexity,
            ill_nested
    );

    PyObject *list = PyList_New(0);
    goal->callback_full([&] (const ArgmaxDiscChartValue& v){
        const int i = std::get<0>(v.cst);
        const int k = std::get<1>(v.cst);
        const int l = std::get<2>(v.cst);
        const int j = std::get<3>(v.cst);
        PyObject *tuple = PyTuple_New(4);
        PyTuple_SET_ITEM(tuple, 0, PyLong_FromLong(i));
        PyTuple_SET_ITEM(tuple, 1, PyLong_FromLong(k));
        PyTuple_SET_ITEM(tuple, 2, PyLong_FromLong(l));
        PyTuple_SET_ITEM(tuple, 3, PyLong_FromLong(j));

        PyList_Append(list, tuple);
        Py_DECREF(tuple);
    }, false);

    return list;
}

PyObject* argmax_disc_as_list(
        ArgmaxDiscChart* chart,
        const unsigned size,
        const unsigned n_labels,
        const unsigned n_disc_labels,
        const float* const weights_cont,
        const float* const weights_disc,
        const float* const weights_gap,
        const SpanMapInt* gold,
        const unsigned complexity,
        const bool ill_nested
)
{
    // inline const ArgmaxDiscChartValue& argmax_disc(ArgmaxDiscChart& chart, const unsigned size, Op Weights, const unsigned complexity=6, const bool ill_nested=true)
    const ArgmaxDiscChartValue* goal;

    goal = &argmax_disc(
            *chart,
            size,
            [&] (const int i, const int k, const int l, const int j) -> std::pair<float, int>
            {
                return max_weight_label_pair(
                        // cannot pass the function directly because it is inlined
                        [&](const float* const weights, const unsigned size, const unsigned n_labels, const int i, const int j, const int label) -> float
                        {
                            return torch_get_labeled_cst_weight(weights, size, n_labels, i, j, label);
                        },
                        size,
                        n_labels,
                        n_disc_labels,
                        weights_cont, weights_disc, weights_gap,
                        i, k, l, j,
                        gold
                );
            },
            complexity,
            ill_nested
    );

    PyObject *list = PyList_New(0);
    goal->callback_full([&] (const ArgmaxDiscChartValue& v){
        const int i = std::get<0>(v.cst);
        const int k = std::get<1>(v.cst);
        const int l = std::get<2>(v.cst);
        const int j = std::get<3>(v.cst);
        PyObject *tuple = PyTuple_New(5);
        PyTuple_SET_ITEM(tuple, 0, PyLong_FromLong(v.label));
        PyTuple_SET_ITEM(tuple, 1, PyLong_FromLong(i));
        PyTuple_SET_ITEM(tuple, 2, PyLong_FromLong(k));
        PyTuple_SET_ITEM(tuple, 3, PyLong_FromLong(l));
        PyTuple_SET_ITEM(tuple, 4, PyLong_FromLong(j));

        PyList_Append(list, tuple);
        Py_DECREF(tuple);
    }, false);

    return list;
}


PyObject* argmax_disc_as_sentence_ptr(
        ArgmaxDiscChart* chart,
        PyObject* py_sentence,
        PyObject* py_label_id_to_string,
        PyObject* py_disc_label_id_to_string,
        const float* weights_cont,
        const float* weights_disc,
        const float* weights_gap,
        const unsigned complexity,
        const bool ill_nested
)
{
    // set size and n_labels
    PyObject* py_words = PyDict_GetItemString(py_sentence, "words");
    PyObject* py_tags = PyDict_GetItemString(py_sentence, "tags");
    int size = PyList_Size(py_words);
    int n_labels = PyList_Size(py_label_id_to_string);
    int n_disc_labels = PyList_Size(py_disc_label_id_to_string);

    // extract id
    std::string id(PyUnicode_AsUTF8(PyDict_GetItemString(py_sentence, "id")));

    // inline const ArgmaxDiscChartValue& argmax_disc(ArgmaxDiscChart& chart, const unsigned size, Op Weights, const unsigned complexity=6, const bool ill_nested=true)
    const ArgmaxDiscChartValue* goal;

    goal = &argmax_disc(
            *chart,
            size,
            [&] (const int i, const int k, const int l, const int j) -> std::pair<float, int>
            {
                return max_weight_label_pair(
                        // cannot pass the function directly because it is inlined
                        [&](const float* const weights, const unsigned size, const unsigned n_labels, const int i, const int j, const int label) -> float
                        {
                            return torch_get_labeled_cst_weight(weights, size, n_labels, i, j, label);
                        },
                        size,
                        n_labels,
                        n_disc_labels,
                        weights_cont, weights_disc, weights_gap,
                        i, k, l, j
                );
            },
            complexity,
            ill_nested
    );

    const auto root = reconstruct(*goal, [&] (const ChartCstDisc& cst, const int label_id) -> std::string
                                  {
                                      if (label_id < 0)
                                          return "NULL";
                                      else
                                      {
                                          if (std::get<1>(cst) < 0)
                                          {
                                              std::string s(PyUnicode_AsUTF8(PyList_GetItem(py_label_id_to_string, label_id)));
                                              return s;
                                          }
                                          else
                                          {
                                              std::string s(PyUnicode_AsUTF8(PyList_GetItem(py_disc_label_id_to_string, label_id)));
                                              return s;
                                          }
                                      }
                                  }
    );

    std::vector<Token> tokens;
    for (int i = 0 ; i < size ; ++ i)
    {
        std::string word(PyUnicode_AsUTF8(PyList_GetItem(py_words, i)));
        std::string tag(PyUnicode_AsUTF8(PyList_GetItem(py_tags, i)));

        tokens.emplace_back(word, tag, true);
    }

    Sentence* sentence = new Sentence(
            id,
            tokens.begin(), tokens.end(),
            root
    );

    return PyLong_FromVoidPtr(sentence);
}


PyObject* argmax_cubic_as_sentence_ptr(
        ArgmaxCubicChart* chart,
        PyObject* py_sentence,
        PyObject* py_label_id_to_string,
        PyObject* py_disc_label_id_to_string,
        const float* weights_cont,
        const float* weights_disc,
        const float* weights_gap
)
{
    // set size and n_labels
    PyObject* py_words = PyDict_GetItemString(py_sentence, "words");
    PyObject* py_tags = PyDict_GetItemString(py_sentence, "tags");
    int size = PyList_Size(py_words);
    int n_labels = PyList_Size(py_label_id_to_string);
    int n_disc_labels = PyList_Size(py_disc_label_id_to_string);

    // extract id
    std::string id(PyUnicode_AsUTF8(PyDict_GetItemString(py_sentence, "id")));

    const ArgmaxCubicChartValue* goal;

    goal = &argmax_cubic(
            *chart,
            size,
            n_disc_labels,
            [&] (const int i, const int j) -> std::pair<float, int>
            {
                return max_weight_label_pair(
                        // cannot pass the function directly because it is inlined
                        [&](const float* const weights, const unsigned size, const unsigned n_labels, const int i, const int j, const int label) -> float
                        {
                            return torch_get_labeled_cst_weight(weights, size, n_labels, i, j, label);
                        },
                        size,
                        n_labels,
                        n_disc_labels,
                        weights_cont,
                        weights_disc, weights_gap, // they won't be used but pass anyway
                        i, -1, -1, j
                );
            },
            // span
            [&] (const int i, const int j, const int label) -> float
            {
                return torch_get_labeled_cst_weight(weights_disc, size, n_disc_labels, i, j, label);
            },
            // gap
            [&] (const int i, const int j, const int label) -> float
            {
                return torch_get_labeled_cst_weight(weights_gap, size, n_disc_labels, i+1, j-1, label);
            }
    );

    const auto root = reconstruct_cubic(*goal, [&] (const int label_id, bool cont) -> std::string
                                  {
                                      {
                                          if (cont)
                                          {
                                              if (label_id < 0)
                                                  return "NULL";
                                              else
                                                  {
                                                  std::string s(PyUnicode_AsUTF8(
                                                          PyList_GetItem(py_label_id_to_string, label_id)));
                                                  return s;
                                              }
                                          }
                                          else
                                          {
                                              if (label_id < 0)
                                                  throw std::runtime_error("Negative label id");
                                              std::string s(PyUnicode_AsUTF8(PyList_GetItem(py_disc_label_id_to_string, label_id)));
                                              return s;
                                          }
                                      }
                                  }
    );

    std::vector<Token> tokens;
    for (int i = 0 ; i < size ; ++ i)
    {
        std::string word(PyUnicode_AsUTF8(PyList_GetItem(py_words, i)));
        std::string tag(PyUnicode_AsUTF8(PyList_GetItem(py_tags, i)));

        tokens.emplace_back(word, tag, true);
    }

    Sentence* sentence = new Sentence(
            id,
            tokens.begin(), tokens.end(),
            root
    );

    return PyLong_FromVoidPtr(sentence);
}
