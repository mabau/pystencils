import numpy as np

from pystencils import get_code_obj
from pystencils.astnodes import Block, KernelFunction, SympyAssignment
from pystencils.cpu import make_python_function
from pystencils.field import Field
from pystencils.enums import Target, Backend
from pystencils.transformations import (
    make_loop_over_domain, move_constants_before_loop, resolve_field_accesses)


def test_jacobi_fixed_field_size():
    size = (30, 20)

    src_field_c = np.random.rand(*size)
    src_field_py = np.copy(src_field_c)
    dst_field_c = np.zeros(size)
    dst_field_py = np.zeros(size)

    f = Field.create_from_numpy_array("f", src_field_c)
    d = Field.create_from_numpy_array("d", dst_field_c)

    jacobi = SympyAssignment(d[0, 0], (f[1, 0] + f[-1, 0] + f[0, 1] + f[0, -1]) / 4)
    body = Block([jacobi])
    loop_node, gl_info = make_loop_over_domain(body)
    ast_node = KernelFunction(loop_node, Target.CPU, Backend.C, make_python_function, ghost_layers=gl_info)
    resolve_field_accesses(ast_node)
    move_constants_before_loop(ast_node)

    for x in range(1, size[0] - 1):
        for y in range(1, size[1] - 1):
            dst_field_py[x, y] = 0.25 * (src_field_py[x - 1, y] + src_field_py[x + 1, y] +
                                         src_field_py[x, y - 1] + src_field_py[x, y + 1])

    kernel = ast_node.compile()
    kernel(f=src_field_c, d=dst_field_c)
    error = np.sum(np.abs(dst_field_py - dst_field_c))
    np.testing.assert_allclose(error, 0.0, atol=1e-13)

    code_display = get_code_obj(ast_node)
    assert 'for' in str(code_display)
    assert 'for' in code_display._repr_html_()


def test_jacobi_variable_field_size():
    size = (3, 3, 3)
    f = Field.create_generic("f", 3)
    d = Field.create_generic("d", 3)
    jacobi = SympyAssignment(d[0, 0, 0], (f[1, 0, 0] + f[-1, 0, 0] + f[0, 1, 0] + f[0, -1, 0]) / 4)
    body = Block([jacobi])
    loop_node, gl_info = make_loop_over_domain(body)
    ast_node = KernelFunction(loop_node, Target.CPU, Backend.C, make_python_function, ghost_layers=gl_info)
    resolve_field_accesses(ast_node)
    move_constants_before_loop(ast_node)

    src_field_c = np.random.rand(*size)
    src_field_py = np.copy(src_field_c)
    dst_field_c = np.zeros(size)
    dst_field_py = np.zeros(size)

    for x in range(1, size[0] - 1):
        for y in range(1, size[1] - 1):
            for z in range(1, size[2] - 1):
                dst_field_py[x, y, z] = 0.25 * (src_field_py[x - 1, y, z] + src_field_py[x + 1, y, z] +
                                                src_field_py[x, y - 1, z] + src_field_py[x, y + 1, z])

    kernel = ast_node.compile()
    kernel(f=src_field_c, d=dst_field_c)
    error = np.sum(np.abs(dst_field_py - dst_field_c))
    np.testing.assert_allclose(error, 0.0, atol=1e-13)
