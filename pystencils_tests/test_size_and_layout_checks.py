import numpy as np
import pytest

import pystencils
import sympy as sp

from pystencils import Assignment, Field, create_kernel, fields


def test_size_check():
    """Kernel with two fixed-sized fields creating with same size but calling with wrong size"""
    src = np.zeros((20, 21, 9))
    dst = np.zeros_like(src)

    sym_src = Field.create_from_numpy_array("src", src, index_dimensions=1)
    sym_dst = Field.create_from_numpy_array("dst", dst, index_dimensions=1)
    update_rule = Assignment(sym_dst(0), sym_src[-1, 1](1) + sym_src[1, -1](2))
    ast = create_kernel([update_rule])
    func = ast.compile()

    # change size of src field
    new_shape = [a - 7 for a in src.shape]
    src = np.zeros(new_shape)
    dst = np.zeros(new_shape)

    with pytest.raises(ValueError) as e:
        func(src=src, dst=dst)
    assert 'Wrong shape' in str(e.value)


def test_fixed_size_mismatch_check():
    """Create kernel with two differently sized but constant fields """
    src = np.zeros((20, 21, 9))
    dst = np.zeros((21, 21, 9))

    sym_src = Field.create_from_numpy_array("src", src, index_dimensions=1)
    sym_dst = Field.create_from_numpy_array("dst", dst, index_dimensions=1)
    update_rule = Assignment(sym_dst(0),
                             sym_src[-1, 1](1) + sym_src[1, -1](2))

    with pytest.raises(ValueError) as e:
        create_kernel([update_rule])
    assert 'Differently sized field accesses' in str(e.value)


def test_fixed_and_variable_field_check():
    """Create kernel with two variable sized fields - calling them with different sizes"""
    src = np.zeros((20, 21, 9))

    sym_src = Field.create_from_numpy_array("src", src, index_dimensions=1)
    sym_dst = Field.create_generic("dst", spatial_dimensions=2, index_dimensions=1)

    update_rule = Assignment(sym_dst(0),
                             sym_src[-1, 1](1) + sym_src[1, -1](2))

    with pytest.raises(ValueError) as e:
        create_kernel(update_rule)
    assert 'Mixing fixed-shaped and variable-shape fields' in str(e.value)


def test_two_variable_shaped_fields():
    src = np.zeros((20, 21, 9))
    dst = np.zeros((22, 21, 9))

    sym_src = Field.create_generic("src", spatial_dimensions=2, index_dimensions=1)
    sym_dst = Field.create_generic("dst", spatial_dimensions=2, index_dimensions=1)
    update_rule = Assignment(sym_dst(0),
                             sym_src[-1, 1](1) + sym_src[1, -1](2))

    ast = create_kernel([update_rule])
    func = ast.compile()

    with pytest.raises(TypeError) as e:
        func(src=src, dst=dst)
    assert 'must have same' in str(e.value)


def test_ssa_checks():
    f, g = fields("f, g : double[2D]")
    a, b, c = sp.symbols("a b c")

    with pytest.raises(ValueError) as e:
        create_kernel([Assignment(c, f[0, 1]),
                       Assignment(c, f[1, 0]),
                       Assignment(g[0, 0], c)])
    assert 'Assignments not in SSA form' in str(e.value)

    with pytest.raises(ValueError) as e:
        create_kernel([Assignment(c, a + 3),
                       Assignment(a, 42),
                       Assignment(g[0, 0], c)])
    assert 'Symbol a is written, after it has been read' in str(e.value)

    with pytest.raises(ValueError) as e:
        create_kernel([Assignment(c, c + 1),
                       Assignment(g[0, 0], c)])
    assert 'Symbol c is written, after it has been read' in str(e.value)


def test_loop_independence_checks():
    f, g = fields("f, g : double[2D]")
    v = fields("v(2) : double[2D]")

    with pytest.raises(ValueError) as e:
        create_kernel([Assignment(g[0, 1], f[0, 1]),
                       Assignment(g[0, 0], f[1, 0])])
    assert 'Field g is written at two different locations' in str(e.value)

    # This is not allowed - because this is not SSA (it can be overwritten with allow_double_writes)
    with pytest.raises(ValueError) as e:
        create_kernel([Assignment(g[0, 2], f[0, 1]),
                       Assignment(g[0, 2], 2 * g[0, 2])])

    # This is allowed - because allow_double_writes is True now
    create_kernel([Assignment(g[0, 2], f[0, 1]),
                   Assignment(g[0, 2], 2 * g[0, 2])],
                  config=pystencils.CreateKernelConfig(allow_double_writes=True))

    with pytest.raises(ValueError) as e:
        create_kernel([Assignment(v[0, 2](1), f[0, 1]),
                       Assignment(v[0, 1](0), 4),
                       Assignment(v[0, 2](1), 2 * v[0, 2](1))])

    with pytest.raises(ValueError) as e:
        create_kernel([Assignment(g[0, 1], 3),
                       Assignment(f[0, 1], 2 * g[0, 2])])
    assert 'Field g is read at (0, 2) and written at (0, 1)' in str(e.value)
