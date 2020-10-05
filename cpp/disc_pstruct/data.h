#pragma once

#include <Python.h>
#include "disc_pstruct/argmax-disc/chart.h"

PyObject* read_data(PyObject* py_path, const bool fix_parenthesis);
void write_data(PyObject* py_list, const bool fix_parenthesis);
