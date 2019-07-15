import numpy as np

import pystencils as ps


def test_blocking_staggered():
    f, stag = ps.fields("f, stag(3): double[3D]")
    terms = [
       f[0, 0, 0] - f[-1, 0, 0],
       f[0, 0, 0] - f[0, -1, 0],
       f[0, 0, 0] - f[0, 0, -1],
    ]
    kernel = ps.create_staggered_kernel(stag, terms, cpu_blocking=(3, 16, 8)).compile()
    reference_kernel = ps.create_staggered_kernel(stag, terms).compile()
    print(ps.show_code(kernel.ast))

    f_arr = np.random.rand(80, 33, 19)
    stag_arr = np.zeros((80, 33, 19, 3))
    stag_ref = np.zeros((80, 33, 19, 3))
    kernel(f=f_arr, stag=stag_arr)
    reference_kernel(f=f_arr, stag=stag_ref)
    np.testing.assert_almost_equal(stag_arr, stag_ref)
