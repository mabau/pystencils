import numpy as np
import pytest

from pystencils import Assignment, Field
try:
    from pystencils.llvm import create_kernel, make_python_function
except ModuleNotFoundError:
    pytest.importorskip("llvmlite")


def test_size_check():
    """Kernel with two fixed-sized fields creating with same size but calling with wrong size"""
    src = np.zeros((20, 21, 9))
    dst = np.zeros_like(src)

    sym_src = Field.create_from_numpy_array("src", src, index_dimensions=1)
    sym_dst = Field.create_from_numpy_array("dst", dst, index_dimensions=1)
    update_rule = Assignment(sym_dst(0),
                             sym_src[-1, 1](1) + sym_src[1, -1](2))
    ast = create_kernel([update_rule])
    func = make_python_function(ast)

    # change size of src field
    new_shape = [a - 7 for a in src.shape]
    src = np.zeros(new_shape)
    dst = np.zeros(new_shape)

    try:
        func(src=src, dst=dst)
        assert False, "Expected ValueError because fields with different sized where passed"
    except ValueError:
        pass


def test_fixed_size_mismatch_check():
    """Create kernel with two differently sized but constant fields """
    src = np.zeros((20, 21, 9))
    dst = np.zeros((21, 21, 9))

    sym_src = Field.create_from_numpy_array("src", src, index_dimensions=1)
    sym_dst = Field.create_from_numpy_array("dst", dst, index_dimensions=1)
    update_rule = Assignment(sym_dst(0),
                             sym_src[-1, 1](1) + sym_src[1, -1](2))

    try:
        create_kernel([update_rule])
        assert False, "Expected ValueError because fields with different sized where passed"
    except ValueError:
        pass


def test_fixed_and_variable_field_check():
    """Create kernel with two variable sized fields - calling them with different sizes"""
    src = np.zeros((20, 21, 9))

    sym_src = Field.create_from_numpy_array("src", src, index_dimensions=1)
    sym_dst = Field.create_generic("dst", spatial_dimensions=2, index_dimensions=1)

    update_rule = Assignment(sym_dst(0),
                             sym_src[-1, 1](1) + sym_src[1, -1](2))

    try:
        create_kernel([update_rule])
        assert False, "Expected ValueError because fields with different sized where passed"
    except ValueError:
        pass


def test_two_variable_shaped_fields():
    src = np.zeros((20, 21, 9))
    dst = np.zeros((22, 21, 9))

    sym_src = Field.create_generic("src", spatial_dimensions=2, index_dimensions=1)
    sym_dst = Field.create_generic("dst", spatial_dimensions=2, index_dimensions=1)
    update_rule = Assignment(sym_dst(0),
                             sym_src[-1, 1](1) + sym_src[1, -1](2))

    ast = create_kernel([update_rule])
    func = make_python_function(ast)

    try:
        func(src=src, dst=dst)
        assert False, "Expected ValueError because fields with different sized where passed"
    except ValueError:
        pass
