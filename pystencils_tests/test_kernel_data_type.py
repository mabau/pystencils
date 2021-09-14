from collections import defaultdict

import numpy as np
import pytest
from sympy.abc import x, y

from pystencils import Assignment, create_kernel, fields, CreateKernelConfig
from pystencils.transformations import adjust_c_single_precision_type


@pytest.mark.parametrize("data_type", ("float", "double"))
def test_single_precision(data_type):
    dtype = f"float{64 if data_type == 'double' else 32}"
    s = fields(f"s: {dtype}[1D]")
    assignments = [Assignment(x, y), Assignment(s[0], x)]
    ast = create_kernel(assignments, config=CreateKernelConfig(data_type=data_type))
    assert ast.body.args[0].lhs.dtype.numpy_dtype == np.dtype(dtype)
    assert ast.body.args[0].rhs.dtype.numpy_dtype == np.dtype(dtype)
    assert ast.body.args[1].body.args[0].rhs.dtype.numpy_dtype == np.dtype(dtype)


def test_adjustment_dict():
    d = dict({"x": "float", "y": "double"})
    adjust_c_single_precision_type(d)
    assert np.dtype(d["x"]) == np.dtype("float32")
    assert np.dtype(d["y"]) == np.dtype("float64")


def test_adjustement_default_dict():
    dd = defaultdict(lambda: "float")
    dd["x"]
    adjust_c_single_precision_type(dd)
    dd["y"]
    assert np.dtype(dd["x"]) == np.dtype("float32")
    assert np.dtype(dd["y"]) == np.dtype("float32")
    assert np.dtype(dd["z"]) == np.dtype("float32")
