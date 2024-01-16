import numpy as np

import pystencils as ps


def test_blocking_staggered():
    f = ps.fields("f: double[3D]")
    stag = ps.fields("stag(3): double[3D]", field_type=ps.FieldType.STAGGERED)
    terms = [
       f[0, 0, 0] - f[-1, 0, 0],
       f[0, 0, 0] - f[0, -1, 0],
       f[0, 0, 0] - f[0, 0, -1],
    ]
    assignments = [ps.Assignment(stag.staggered_access(d), terms[i]) for i, d in enumerate(stag.staggered_stencil)]
    reference_kernel = ps.create_staggered_kernel(assignments)
    print(ps.show_code(reference_kernel))
    reference_kernel = reference_kernel.compile()
    kernel = ps.create_staggered_kernel(assignments, cpu_blocking=(3, 16, 8)).compile()
    print(ps.show_code(kernel.ast))

    f_arr = np.random.rand(80, 33, 19)
    stag_arr = np.zeros((80, 33, 19, 3))
    stag_ref = np.zeros((80, 33, 19, 3))
    kernel(f=f_arr, stag=stag_arr)
    reference_kernel(f=f_arr, stag=stag_ref)
    np.testing.assert_almost_equal(stag_arr, stag_ref)
