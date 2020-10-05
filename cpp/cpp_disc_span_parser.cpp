#include "cpp_disc_span_parser.h"
#include "disc_pstruct/argmax-disc/algorithms.h"
#include "disc_pstruct/argmax-disc/chart.h"
#include "disc_pstruct/data.h"
#include "utils/torch-indexing.h"

extern "C"
{
static PyMethodDef python_methods[] =
        {
                {"max_recall", py_max_recall, METH_VARARGS, "Compute max recall"},
                {"new_ArgmaxDiscChart", py_new_ArgmaxDiscChart, METH_VARARGS, "Create char for argmax"},
                {"new_ArgmaxDiscChart_parallel", py_new_ArgmaxDiscChart_parallel, METH_VARARGS, "Create char for argmax"},
                {"delete_ArgmaxDiscChart", py_delete_ArgmaxDiscChart, METH_VARARGS, "Delete char for argmax"},
                {"delete_ArgmaxDiscChart_parallel", py_delete_ArgmaxDiscChart_parallel, METH_VARARGS, "Delete char for argmax"},
                {"argmax_disc_as_list", py_argmax_disc_as_list, METH_VARARGS, "Argmax"},
                {"argmax_disc_as_list_with_gold_spans", py_argmax_disc_as_list_with_gold_spans, METH_VARARGS, "Argmax"},
                {"argmax_disc_as_list_parallel", py_argmax_disc_as_list_parallel, METH_VARARGS, "Argmax"},
                {"argmax_disc_as_sentence_ptr", py_argmax_disc_as_sentence_ptr, METH_VARARGS, "Argmax"},
                {"read_data", py_read_data, METH_VARARGS, "Read data"},
                {"write_data", py_write_data, METH_VARARGS, "Write data"},
                {"build_span_map_int", py_build_span_map_int, METH_VARARGS, "Read data"},
                // cubic
                {"new_ArgmaxCubicChart", py_new_ArgmaxCubicChart, METH_VARARGS, "Create char for argmax cubic"},
                {"argmax_cubic_as_sentence_ptr", py_argmax_cubic_as_sentence_ptr, METH_VARARGS, "Argmax Cubic"},
                {NULL, NULL, 0, NULL}
        };

static struct PyModuleDef cModPyDem =
        {
                PyModuleDef_HEAD_INIT,
                "cpp_disc_span_parser",
                NULL,
                -1,
                python_methods
        };
}

PyMODINIT_FUNC
PyInit_cpp_disc_span_parser(void)
{
    return PyModule_Create(&cModPyDem);
}

PyObject* py_max_recall(PyObject*, PyObject* args)
{
    PyObject* py_chart;
    long size;
    PyObject* py_ptr; // gold spans
    long complexity;
    PyObject* py_illnested;

    if (!PyArg_ParseTuple(args, "OlOlO", &py_chart, &size, &py_ptr, &complexity, &py_illnested))
        return NULL;

    bool ill_nested = PyObject_IsTrue(py_illnested);
    ArgmaxDiscChart* chart = (ArgmaxDiscChart*) PyLong_AsVoidPtr(py_chart);
    if (chart == NULL)
        return NULL;
    SpanMapInt* gold_spans = (SpanMapInt*) PyLong_AsVoidPtr(py_ptr);
    if (gold_spans == NULL)
        return NULL;

    return max_recall(chart, size, gold_spans, (unsigned) complexity, ill_nested);
}

PyObject* py_build_span_map_int(PyObject*, PyObject* args)
{
    PyObject* py_spans;
    SpanMapInt* span_map = new SpanMapInt();
    if (!PyArg_ParseTuple(args, "O", &py_spans))
        return NULL;

    if (not PyList_Check(py_spans))
    {
        std::cerr << "Object passed is not a list";
        return NULL;
    }
    auto n_spans = PyList_Size(py_spans);
    for (int span_indice = 0 ; span_indice < n_spans ; ++span_indice)
    {
        PyObject* py_span = PyList_GetItem(py_spans, span_indice);
        if (not PyTuple_Check(py_span))
        {
            std::cerr << "Object passed is not a list of tuples";
            return NULL;
        }
        auto size = PyTuple_Size(py_span);
        if (size != 5)
        {
            std::cerr << "Tuple is not of size 5";
            return NULL;
        }

        // label, i, k, l, j
        PyObject* py_label = PyTuple_GetItem(py_span, 0);
        PyObject* py_i = PyTuple_GetItem(py_span, 1);
        PyObject* py_k = PyTuple_GetItem(py_span, 2);
        PyObject* py_l = PyTuple_GetItem(py_span, 3);
        PyObject* py_j = PyTuple_GetItem(py_span, 4);
        int label = PyLong_AsLong(py_label);
        int i = PyLong_AsLong(py_i);
        int k = PyLong_AsLong(py_k);
        int l = PyLong_AsLong(py_l);
        int j = PyLong_AsLong(py_j);
        span_map->emplace(std::make_tuple(i, k, l, j), label);
    }
    return PyLong_FromVoidPtr(span_map);
}

PyObject* py_new_ArgmaxDiscChart(PyObject*, PyObject* args)
{
    long size;
    long n_charts;
    if (!PyArg_ParseTuple(args, "l", &size, &n_charts))
        return NULL;

    return PyLong_FromVoidPtr((void*) new ArgmaxDiscChart(size));
}

PyObject* py_delete_ArgmaxDiscChart(PyObject*, PyObject* args)
{
    PyObject* py_chart = nullptr;

    if (!PyArg_ParseTuple(args, "O", py_chart))
        return NULL;

    ArgmaxDiscChart* chart = (ArgmaxDiscChart*) PyLong_AsVoidPtr(py_chart);
    if (chart == NULL)
        return NULL;

    delete chart;
    return Py_None;
}

PyObject* py_new_ArgmaxCubicChart(PyObject*, PyObject* args)
{
    long size;
    long n_labels;
    if (!PyArg_ParseTuple(args, "ll", &size, &n_labels))
        return NULL;

    return PyLong_FromVoidPtr((void*) new ArgmaxCubicChart(size, n_labels));
}

PyObject* py_new_ArgmaxDiscChart_parallel(PyObject*, PyObject* args)
{
    long size;
    long n_charts;
    if (!PyArg_ParseTuple(args, "ll", &size, &n_charts))
        return NULL;

    std::vector<ArgmaxDiscChart*>* charts = new std::vector<ArgmaxDiscChart*>();
    for (int i = 0 ; i < n_charts ; ++i)
        charts->emplace_back(new ArgmaxDiscChart(size));
    return PyLong_FromVoidPtr((void*) charts);
}

PyObject* py_delete_ArgmaxDiscChart_parallel(PyObject*, PyObject* args)
{
    PyObject* py_chart = nullptr;

    if (!PyArg_ParseTuple(args, "O", py_chart))
        return NULL;

    auto chart = (std::vector<ArgmaxDiscChart*>*) PyLong_AsVoidPtr(py_chart);
    for (unsigned i = 0 ; i < chart->size() ; ++i)
        delete chart->at(i);
    delete chart;

    return Py_None;
}

PyObject* py_read_data(PyObject*, PyObject* args)
{
    PyObject* path;
    int fix_parenthesis;

    if (!PyArg_ParseTuple(args, "O", &path, &fix_parenthesis))
        return NULL;

    return read_data(path, false);
}

PyObject* py_write_data(PyObject*, PyObject* args)
{
    PyObject* data;

    if (!PyArg_ParseTuple(args, "O", &data))
        return NULL;

    write_data(data, false);
    return Py_None;
}


PyObject* py_argmax_disc_as_list(PyObject*, PyObject* args)
{
    PyObject* py_chart;
    long size;
    long n_labels;
    long n_disc_labels;
    PyObject* py_cont_weights;
    PyObject* py_disc_weights;
    PyObject* py_gap_weights;
    long complexity;
    PyObject* py_illnested;

    if (!PyArg_ParseTuple(args, "OlllOOOlO", &py_chart, &size, &n_labels, &n_disc_labels, &py_cont_weights, &py_disc_weights, &py_gap_weights, &complexity, &py_illnested))
        return NULL;

    bool ill_nested = PyObject_IsTrue(py_illnested);

    ArgmaxDiscChart* chart = (ArgmaxDiscChart*) PyLong_AsVoidPtr(py_chart);
    if (chart == NULL)
        return NULL;
    const float* const cont_weights = (const float*) PyLong_AsVoidPtr(py_cont_weights);
    if (cont_weights == NULL)
        return NULL;
    const float* const disc_weights = (const float*) PyLong_AsVoidPtr(py_disc_weights);
    if (disc_weights == NULL)
        return NULL;
    const float* const gap_weights = (const float*) PyLong_AsVoidPtr(py_gap_weights);
    if (gap_weights == NULL)
        return NULL;
    return argmax_disc_as_list(
            chart,
            size,
            n_labels,
            n_disc_labels,
            cont_weights,
            disc_weights,
            gap_weights,
            nullptr,
            complexity,
            ill_nested
    );
}


PyObject* py_argmax_disc_as_list_parallel(PyObject*, PyObject* args)
{
    PyObject* py_chart;
    PyObject* py_sizes;
    PyObject* py_n_labels;
    PyObject* py_n_disc_labels;
    PyObject* py_cont_weights;
    PyObject* py_disc_weights;
    PyObject* py_gap_weights;
    PyObject* py_gold_spans;
    long complexity;
    PyObject* py_illnested;

    if (!PyArg_ParseTuple(args, "OOOOOOOOlO", &py_chart, &py_sizes, &py_n_labels, &py_n_disc_labels, &py_cont_weights, &py_disc_weights, &py_gap_weights, &py_gold_spans, &complexity, &py_illnested))
        return NULL;
    bool ill_nested = PyObject_IsTrue(py_illnested);

    std::vector<ArgmaxDiscChart*>* charts = (std::vector<ArgmaxDiscChart*>*) PyLong_AsVoidPtr(py_chart);
    const auto n_inputs = PyList_Size(py_cont_weights);
    const auto n_charts = charts->size();
    std::vector<std::vector<std::tuple<int,int,int,int,int>>> pred_spans(n_inputs);

    #pragma omp parallel for
    for (unsigned chart_index = 0 ; chart_index < n_charts ; ++chart_index)
    {
        ArgmaxDiscChart* chart = charts->at(chart_index);
        for (unsigned input_index = chart_index ; input_index < n_inputs ; input_index += n_charts)
        {
            const unsigned size = PyLong_AsUnsignedLong(PyList_GetItem(py_sizes, input_index));
            const unsigned n_labels = PyLong_AsUnsignedLong(PyList_GetItem(py_n_labels, input_index));
            const unsigned n_disc_labels = PyLong_AsUnsignedLong(PyList_GetItem(py_n_disc_labels, input_index));
            auto* const weights_cont = (const float* const) PyLong_AsVoidPtr(PyList_GetItem(py_cont_weights, input_index));
            auto* const weights_disc = (const float* const) PyLong_AsVoidPtr(PyList_GetItem(py_disc_weights, input_index));
            auto* const weights_gap = (const float* const) PyLong_AsVoidPtr(PyList_GetItem(py_gap_weights, input_index));
            const SpanMapInt* gold = nullptr;
            if (py_gold_spans != Py_None)
                gold = (const SpanMapInt*) PyLong_AsVoidPtr(PyList_GetItem(py_gold_spans, input_index));

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
            goal->callback_full([&] (const ArgmaxDiscChartValue& v){
                const int label = v.label;
                const int i = std::get<0>(v.cst);
                const int k = std::get<1>(v.cst);
                const int l = std::get<2>(v.cst);
                const int j = std::get<3>(v.cst);
                pred_spans.at(input_index).emplace_back(label, i, k, l, j);
            }, false);
        }
    }

    PyObject *ret = PyList_New(0);
    for (unsigned input_index = 0 ; input_index < n_inputs ; ++input_index)
    {
        PyObject *list = PyList_New(0);
        for (const auto& span : pred_spans.at(input_index))
        {
            const int label = std::get<0>(span);
            const int i = std::get<1>(span);
            const int k = std::get<2>(span);
            const int l = std::get<3>(span);
            const int j = std::get<4>(span);
            PyObject *tuple = PyTuple_New(5);
            PyTuple_SET_ITEM(tuple, 0, PyLong_FromLong(label));
            PyTuple_SET_ITEM(tuple, 1, PyLong_FromLong(i));
            PyTuple_SET_ITEM(tuple, 2, PyLong_FromLong(k));
            PyTuple_SET_ITEM(tuple, 3, PyLong_FromLong(l));
            PyTuple_SET_ITEM(tuple, 4, PyLong_FromLong(j));

            PyList_Append(list, tuple);
            Py_DECREF(tuple);
        }
        PyList_Append(ret, list);
        Py_DECREF(list);
    }

    return ret;

}


PyObject* py_argmax_disc_as_list_with_gold_spans(PyObject*, PyObject* args)
{
    PyObject* py_chart;
    long size;
    long n_labels;
    long n_disc_labels;
    PyObject* py_cont_weights;
    PyObject* py_disc_weights;
    PyObject* py_gap_weights;
    PyObject* py_ptr;
    long complexity;
    PyObject* py_illnested;

    if (!PyArg_ParseTuple(args, "OlllOOOOlO", &py_chart, &size, &n_labels, &n_disc_labels, &py_cont_weights, &py_disc_weights, &py_gap_weights, &py_ptr, &complexity, &py_illnested))
        return NULL;
    bool ill_nested = PyObject_IsTrue(py_illnested);

    ArgmaxDiscChart* chart = (ArgmaxDiscChart*) PyLong_AsVoidPtr(py_chart);
    if (chart == NULL)
    {
        std::cerr << "Error with chart\n";
        return NULL;
    }
    const float* const cont_weights = (const float*) PyLong_AsVoidPtr(py_cont_weights);
    if (cont_weights == NULL)
    {
        std::cerr << "Error with cont_weights\n";
        return NULL;
    }
    const float* const disc_weights = (const float*) PyLong_AsVoidPtr(py_disc_weights);
    if (disc_weights == NULL)
    {
        std::cerr << "Error with disc_weights\n";
        return NULL;
    }
    const float* const gap_weights = (const float*) PyLong_AsVoidPtr(py_gap_weights);
    if (gap_weights == NULL)
    {
        std::cerr << "Error with gap_weights\n";
        return NULL;
    }
    SpanMapInt* gold_spans = (SpanMapInt*) PyLong_AsVoidPtr(py_ptr);
    if (gold_spans == NULL)
    {
        std::cerr << "Error with gold_spans\n";
        return NULL;
    }

    return argmax_disc_as_list(
            chart,
            size,
            n_labels,
            n_disc_labels,
            cont_weights,
            disc_weights,
            gap_weights,
            gold_spans,
            complexity,
            ill_nested
    );
}

PyObject* py_argmax_disc_as_sentence_ptr(PyObject*, PyObject* args) {
    PyObject *py_chart;
    PyObject *py_sentence;
    PyObject *py_label_id_to_string;
    PyObject *py_disc_label_id_to_string;
    PyObject *py_cont_weights;
    PyObject *py_disc_weights;
    PyObject *py_gap_weights;
    long complexity;
    PyObject *py_illnested;

    if (!PyArg_ParseTuple(args, "OOOOOOOlO", &py_chart, &py_sentence, &py_label_id_to_string,
                          &py_disc_label_id_to_string, &py_cont_weights, &py_disc_weights, &py_gap_weights, &complexity,
                          &py_illnested))
        return NULL;
    bool ill_nested = PyObject_IsTrue(py_illnested);

    ArgmaxDiscChart *chart = (ArgmaxDiscChart *) PyLong_AsVoidPtr(py_chart);
    if (chart == NULL)
        return NULL;
    const float *const cont_weights = (const float *) PyLong_AsVoidPtr(py_cont_weights);
    if (cont_weights == NULL)
        return NULL;
    const float *const disc_weights = (const float *) PyLong_AsVoidPtr(py_disc_weights);
    if (disc_weights == NULL)
        return NULL;
    const float *const gap_weights = (const float *) PyLong_AsVoidPtr(py_gap_weights);
    if (gap_weights == NULL)
        return NULL;

    return argmax_disc_as_sentence_ptr(
            chart,
            py_sentence,
            py_label_id_to_string,
            py_disc_label_id_to_string,
            cont_weights,
            disc_weights,
            gap_weights,
            complexity,
            ill_nested
    );
}

PyObject* py_argmax_cubic_as_sentence_ptr(PyObject*, PyObject* args)
{
    PyObject* py_chart;
    PyObject* py_sentence;
    PyObject* py_label_id_to_string;
    PyObject* py_disc_label_id_to_string;
    PyObject* py_cont_weights;
    PyObject* py_disc_weights;
    PyObject* py_gap_weights;

    if (!PyArg_ParseTuple(args, "OOOOOOO", &py_chart, &py_sentence, &py_label_id_to_string, &py_disc_label_id_to_string, &py_cont_weights, &py_disc_weights, &py_gap_weights))
        return NULL;

    ArgmaxCubicChart* chart = (ArgmaxCubicChart*) PyLong_AsVoidPtr(py_chart);
    if (chart == NULL)
        return NULL;
    const float* const cont_weights = (const float*) PyLong_AsVoidPtr(py_cont_weights);
    if (cont_weights == NULL)
        return NULL;
    const float* const disc_weights = (const float*) PyLong_AsVoidPtr(py_disc_weights);
    if (disc_weights == NULL)
        return NULL;
    const float* const gap_weights = (const float*) PyLong_AsVoidPtr(py_gap_weights);
    if (gap_weights == NULL)
        return NULL;

    return argmax_cubic_as_sentence_ptr(
            chart,
            py_sentence,
            py_label_id_to_string,
            py_disc_label_id_to_string,
            cont_weights,
            disc_weights,
            gap_weights
    );
}
