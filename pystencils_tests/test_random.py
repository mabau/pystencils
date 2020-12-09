import numpy as np

import pytest

import pystencils as ps
from pystencils.rng import PhiloxFourFloats, PhiloxTwoDoubles, AESNIFourFloats, AESNITwoDoubles, random_symbol


# curand_Philox4x32_10(make_uint4(124, i, j, 0), make_uint2(0, 0))
philox_reference = np.array([[[3576608082, 1252663339, 1987745383,  348040302],
                              [1032407765,  970978240, 2217005168, 2424826293]],
                             [[2958765206, 3725192638, 2623672781, 1373196132],
                              [ 850605163, 1694561295, 3285694973, 2799652583]]])

@pytest.mark.parametrize('target', ('cpu', 'gpu'))
def test_philox_double(target):
    if target == 'gpu':
        pytest.importorskip('pycuda')

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

    x = philox_reference[:,:,0::2]
    y = philox_reference[:,:,1::2]
    z = x ^ y << (53 - 32)
    double_reference = z * 2.**-53 + 2.**-54
    assert(np.allclose(arr, double_reference, rtol=0, atol=np.finfo(np.float64).eps))


@pytest.mark.parametrize('target', ('cpu', 'gpu'))
def test_philox_float(target):
    if target == 'gpu':
        pytest.importorskip('pycuda')

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

    float_reference = philox_reference * 2.**-32 + 2.**-33
    assert(np.allclose(arr, float_reference, rtol=0, atol=np.finfo(np.float32).eps))

def test_aesni_double():
    dh = ps.create_data_handling((2, 2), default_ghost_layers=0, default_target="cpu")
    f = dh.add_array("f", values_per_cell=2)

    dh.fill('f', 42.0)

    aesni_node = AESNITwoDoubles(dh.dim)
    assignments = [aesni_node,
                   ps.Assignment(f(0), aesni_node.result_symbols[0]),
                   ps.Assignment(f(1), aesni_node.result_symbols[1])]
    kernel = ps.create_kernel(assignments, target=dh.default_target).compile()

    dh.all_to_gpu()
    dh.run_kernel(kernel, time_step=124)
    dh.all_to_cpu()

    arr = dh.gather_array('f')
    assert np.logical_and(arr <= 1.0, arr >= 0).all()


def test_aesni_float():
    dh = ps.create_data_handling((2, 2), default_ghost_layers=0, default_target="cpu")
    f = dh.add_array("f", values_per_cell=4)

    dh.fill('f', 42.0)

    aesni_node = AESNIFourFloats(dh.dim)
    assignments = [aesni_node] + [ps.Assignment(f(i), aesni_node.result_symbols[i]) for i in range(4)]
    kernel = ps.create_kernel(assignments, target=dh.default_target).compile()

    dh.all_to_gpu()
    dh.run_kernel(kernel, time_step=124)
    dh.all_to_cpu()
    arr = dh.gather_array('f')
    assert np.logical_and(arr <= 1.0, arr >= 0).all()

def test_staggered():
    """Make sure that the RNG counter can be substituted during loop cutting"""
    dh = ps.create_data_handling((8, 8), default_ghost_layers=0, default_target="cpu")
    j = dh.add_array("j", values_per_cell=dh.dim, field_type=ps.FieldType.STAGGERED_FLUX)
    a = ps.AssignmentCollection([ps.Assignment(j.staggered_access(n), 0) for n in j.staggered_stencil])
    rng_symbol_gen = random_symbol(a.subexpressions, dim=dh.dim)
    a.main_assignments[0] = ps.Assignment(a.main_assignments[0].lhs, next(rng_symbol_gen))
    kernel = ps.create_staggered_kernel(a, target=dh.default_target).compile()
