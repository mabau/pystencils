import numpy as np

import pystencils as ps
from pystencils.cpu.vectorization import get_supported_instruction_sets
from pystencils.cpu.vectorization import replace_inner_stride_with_one, vectorize


def test_basic_kernel():
    for domain_shape in [(4, 5), (3, 4, 5)]:
        dh = ps.create_data_handling(domain_size=domain_shape, periodicity=True)
        assert all(dh.periodicity)

        f = dh.add_array('f', values_per_cell=1)
        tmp = dh.add_array('tmp', values_per_cell=1)

        stencil_2d = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        stencil_3d = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        stencil = stencil_2d if dh.dim == 2 else stencil_3d

        jacobi = ps.Assignment(tmp.center, sum(f.neighbors(stencil)) / len(stencil))
        kernel = ps.create_kernel(jacobi).compile()

        for b in dh.iterate(ghost_layers=1):
            b['f'].fill(42)
        dh.run_kernel(kernel)
        for b in dh.iterate(ghost_layers=0):
            np.testing.assert_equal(b['f'], 42)

        float_seq = [1.0, 2.0, 3.0, 4.0]
        int_seq = [1, 2, 3]
        for op in ('min', 'max', 'sum'):
            assert (dh.reduce_float_sequence(float_seq, op) == float_seq).all()
            assert (dh.reduce_int_sequence(int_seq, op) == int_seq).all()


def test_basic_blocking_staggered():
    f = ps.fields("f: double[2D]")
    stag = ps.fields("stag(2): double[2D]", field_type=ps.FieldType.STAGGERED)
    terms = [
       f[0, 0] - f[-1, 0],
       f[0, 0] - f[0, -1],
    ]
    assignments = [ps.Assignment(stag.staggered_access(d), terms[i]) for i, d in enumerate(stag.staggered_stencil)]
    kernel = ps.create_staggered_kernel(assignments, cpu_blocking=(3, 16)).compile()
    reference_kernel = ps.create_staggered_kernel(assignments).compile()

    f_arr = np.random.rand(80, 33)
    stag_arr = np.zeros((80, 33, 3))
    stag_ref = np.zeros((80, 33, 3))
    kernel(f=f_arr, stag=stag_arr)
    reference_kernel(f=f_arr, stag=stag_ref)
    np.testing.assert_almost_equal(stag_arr, stag_ref)


def test_basic_vectorization():
    supported_instruction_sets = get_supported_instruction_sets()
    if supported_instruction_sets:
        instruction_set = supported_instruction_sets[-1]
    else:
        instruction_set = None

    f, g = ps.fields("f, g : double[2D]")
    update_rule = [ps.Assignment(g[0, 0], f[0, 0] + f[-1, 0] + f[1, 0] + f[0, 1] + f[0, -1] + 42.0)]
    ast = ps.create_kernel(update_rule)

    replace_inner_stride_with_one(ast)
    vectorize(ast, instruction_set=instruction_set)
    func = ast.compile()

    arr = np.ones((23 + 2, 17 + 2)) * 5.0
    dst = np.zeros_like(arr)

    func(g=dst, f=arr)
    np.testing.assert_equal(dst[1:-1, 1:-1], 5 * 5.0 + 42.0)