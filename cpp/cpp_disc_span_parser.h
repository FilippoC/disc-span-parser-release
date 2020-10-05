#pragma once

#include <Python.h>

extern "C"
{
PyObject* py_max_recall(PyObject* self, PyObject* args);
PyObject* py_new_ArgmaxDiscChart(PyObject* self, PyObject* args);
PyObject* py_new_ArgmaxDiscChart_parallel(PyObject* self, PyObject* args);
PyObject* py_delete_ArgmaxDiscChart(PyObject* self, PyObject* args);
PyObject* py_delete_ArgmaxDiscChart_parallel(PyObject* self, PyObject* args);
PyObject* py_argmax_disc_as_list(PyObject* self, PyObject* args);
PyObject* py_argmax_disc_as_list_with_gold_spans(PyObject* self, PyObject* args);
PyObject* py_argmax_disc_as_list_parallel(PyObject* self, PyObject* args);
PyObject* py_argmax_disc_as_sentence_ptr(PyObject* self, PyObject* args);
PyObject* py_read_data(PyObject* self, PyObject* args);
PyObject* py_write_data(PyObject* self, PyObject* args);
PyObject* py_build_span_map_int(PyObject* self, PyObject* args);

// we have just this implementation for the cubic time discontinuous parser
PyObject* py_new_ArgmaxCubicChart(PyObject* self, PyObject* args);
PyObject* py_argmax_cubic_as_sentence_ptr(PyObject* self, PyObject* args);
}