import numpy as np
import sympy as sp

import pytest

import pystencils as ps
import pystencils.astnodes as ast
from pystencils.field import Field, FieldType
from pystencils.astnodes import Conditional, LoopOverCoordinate, SympyAssignment
from pystencils.cpu import create_kernel, make_python_function
from pystencils.kernelcreation import create_staggered_kernel
from pystencils.transformations import (
    cleanup_blocks, cut_loop, move_constants_before_loop, simplify_conditionals)


def offsets_in_plane(normal_plane, offset_int, dimension):
    offset = [0] * dimension
    offset[normal_plane] = offset_int
    result = [tuple(offset)]
    for i in range(dimension):
        if i == normal_plane:
            continue
        lower = offset.copy()
        upper = offset.copy()
        lower[i] -= 1
        upper[i] += 1
        result.append(tuple(lower))
        result.append(tuple(upper))
    return result


# TODO this fails because the condition of the Conditional is not simplified anymore:
# TODO: ---> transformation.simplify_conditionals
# TODO this should be fixed
@pytest.mark.xfail
def test_staggered_iteration():
    dim = 2
    f_arr = np.arange(5**dim).reshape([5]*dim).astype(np.float64)
    s_arr = np.ones([5]*dim + [dim]) * 1234
    s_arr_ref = s_arr.copy()

    fields_fixed = (Field.create_from_numpy_array('f', f_arr),
                    Field.create_from_numpy_array('s', s_arr, index_dimensions=1, field_type=FieldType.STAGGERED))
    fields_var = (Field.create_generic('f', 2),
                  Field.create_generic('s', 2, index_dimensions=1, index_shape=(dim,), field_type=FieldType.STAGGERED))

    for f, s in [fields_var, fields_fixed]:
        # --- Manual
        eqs = []
        counters = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(dim)]
        conditions = [counters[i] < f.shape[i] - 1 for i in range(dim)]
        for d in range(dim):
            eq = SympyAssignment(s(d), sum(f[o] for o in offsets_in_plane(d, 0, dim)) -
                                 sum(f[o] for o in offsets_in_plane(d, -1, dim)))
            cond = sp.And(*[conditions[i] for i in range(dim) if d != i])
            eqs.append(Conditional(cond, eq))
        # TODO: correct type hint
        config = ps.CreateKernelConfig(target=ps.Target.CPU, ghost_layers=[(1, 0), (1, 0), (1, 0)])
        func = ps.create_kernel(eqs, config=config).compile()

        # --- Built-in optimized
        expressions = []
        for d in range(dim):
            expressions.append(sum(f[o] for o in offsets_in_plane(d, 0, dim)) -
                               sum(f[o] for o in offsets_in_plane(d, -1, dim)))
        assignments = [ps.Assignment(s.staggered_access(d), expressions[i]) for i, d in enumerate(s.staggered_stencil)]
        func_optimized = create_staggered_kernel(assignments).compile()
        pytest.importorskip('islpy')
        assert not func_optimized.ast.atoms(Conditional), "Loop cutting optimization did not work"

        func(f=f_arr, s=s_arr_ref)
        func_optimized(f=f_arr, s=s_arr)
        np.testing.assert_almost_equal(s_arr_ref, s_arr)


def test_staggered_iteration_manual():
    dim = 2
    f_arr = np.arange(5**dim).reshape([5]*dim)
    s_arr = np.ones([5]*dim + [dim]) * 1234
    s_arr_ref = s_arr.copy()

    f = Field.create_from_numpy_array('f', f_arr)
    s = Field.create_from_numpy_array('s', s_arr, index_dimensions=1, field_type=FieldType.STAGGERED)

    eqs = []

    counters = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(dim)]
    conditions = [counters[i] < f.shape[i] - 1 for i in range(dim)]
    conditions2 = counters[0] > f.shape[0] + 5

    for d in range(dim):
        eq = SympyAssignment(s(d), sum(f[o] for o in offsets_in_plane(d, 0, dim)) -
                             sum(f[o] for o in offsets_in_plane(d, -1, dim)))
        cond = sp.And(*[conditions[i] for i in range(dim) if d != i])
        eqs.append(Conditional(cond, eq))

    # this conditional should vanish entirely because it is never true
    eq = SympyAssignment(s(0), f[0, 0])
    cond = sp.And(*[conditions2])
    eqs.append(Conditional(cond, eq))

    config = ps.CreateKernelConfig(target=ps.Target.CPU, ghost_layers=[(1, 0), (1, 0), (1, 0)])
    kernel_ast = ps.create_kernel(eqs, config=config)

    func = make_python_function(kernel_ast)
    func(f=f_arr, s=s_arr_ref)

    inner_loop = [n for n in kernel_ast.atoms(ast.LoopOverCoordinate) if n.is_innermost_loop][0]
    cut_loop(inner_loop, [4])
    outer_loop = [n for n in kernel_ast.atoms(ast.LoopOverCoordinate) if n.is_outermost_loop][0]
    cut_loop(outer_loop, [4])

    simplify_conditionals(kernel_ast.body, loop_counter_simplification=True)
    cleanup_blocks(kernel_ast.body)
    move_constants_before_loop(kernel_ast.body)
    cleanup_blocks(kernel_ast.body)

    pytest.importorskip('islpy')
    assert not kernel_ast.atoms(Conditional), "Loop cutting optimization did not work"

    func_optimized = make_python_function(kernel_ast)
    func_optimized(f=f_arr, s=s_arr)
    np.testing.assert_almost_equal(s_arr_ref, s_arr)


def test_staggered_gpu():
    dim = 2
    f = ps.fields(f"f: double[{dim}D]")
    s = ps.fields("s({dim}): double[{dim}D]".format(dim=dim), field_type=FieldType.STAGGERED)
    expressions = [(f[0, 0] + f[-1, 0]) / 2,
                   (f[0, 0] + f[0, -1]) / 2]
    assignments = [ps.Assignment(s.staggered_access(d), expressions[i]) for i, d in enumerate(s.staggered_stencil)]
    kernel_ast = ps.create_staggered_kernel(assignments, target=ps.Target.GPU, gpu_exclusive_conditions=True)
    assert len(kernel_ast.atoms(Conditional)) == 4

    assignments = [ps.Assignment(s.staggered_access(d), expressions[i]) for i, d in enumerate(s.staggered_stencil)]
    kernel_ast = ps.create_staggered_kernel(assignments, target=ps.Target.GPU, gpu_exclusive_conditions=False)
    assert len(kernel_ast.atoms(Conditional)) == 3
