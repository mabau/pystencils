import numpy as np

import pystencils as ps
from pystencils.rng import PhiloxFourFloats, PhiloxTwoDoubles


def test_philox_double():
    for target in ('cpu', 'gpu'):
        dh = ps.create_data_handling((2, 2), default_ghost_layers=0, default_target=target)
        f = dh.add_array("f", values_per_cell=2)

        dh.fill('f', 42.0)

        philox_node = PhiloxTwoDoubles(dh.dim)
        assignments = [philox_node,
                       ps.Assignment(f(0), philox_node.result_symbols[0]),
                       ps.Assignment(f(1), philox_node.result_symbols[1])]
        kernel = ps.create_kernel(assignments, target=dh.default_target).compile()

        dh.all_to_gpu()
        dh.run_kernel(kernel, time_step=124)
        dh.all_to_cpu()

        arr = dh.gather_array('f')
        assert np.logical_and(arr <= 1.0, arr >= 0).all()


def test_philox_float():
    for target in ('cpu', 'gpu'):
        dh = ps.create_data_handling((2, 2), default_ghost_layers=0, default_target=target)
        f = dh.add_array("f", values_per_cell=4)

        dh.fill('f', 42.0)

        philox_node = PhiloxFourFloats(dh.dim)
        assignments = [philox_node] + [ps.Assignment(f(i), philox_node.result_symbols[i]) for i in range(4)]
        kernel = ps.create_kernel(assignments, target=dh.default_target).compile()

        dh.all_to_gpu()
        dh.run_kernel(kernel, time_step=124)
        dh.all_to_cpu()
        arr = dh.gather_array('f')
        assert np.logical_and(arr <= 1.0, arr >= 0).all()
