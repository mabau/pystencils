import numpy as np

import pytest

import pystencils as ps
from pystencils.astnodes import SympyAssignment
from pystencils.node_collection import NodeCollection
from pystencils.rng import PhiloxFourFloats, PhiloxTwoDoubles, AESNIFourFloats, AESNITwoDoubles, random_symbol
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets
from pystencils.cpu.cpujit import get_compiler_config
from pystencils.typing import TypedSymbol
from pystencils.enums import Target

RNGs = {('philox', 'float'): PhiloxFourFloats, ('philox', 'double'): PhiloxTwoDoubles,
        ('aesni', 'float'): AESNIFourFloats, ('aesni', 'double'): AESNITwoDoubles}

instruction_sets = get_supported_instruction_sets()
if get_compiler_config()['os'] == 'windows':
    # skip instruction sets supported by the CPU but not by the compiler
    if 'avx' in instruction_sets and ('/arch:avx2' not in get_compiler_config()['flags'].lower()
                                      and '/arch:avx512' not in get_compiler_config()['flags'].lower()):
        instruction_sets.remove('avx')
    if 'avx512' in instruction_sets and '/arch:avx512' not in get_compiler_config()['flags'].lower():
        instruction_sets.remove('avx512')


@pytest.mark.parametrize('target, rng', ((Target.CPU, 'philox'), (Target.CPU, 'aesni'), (Target.GPU, 'philox')))
@pytest.mark.parametrize('precision', ('float', 'double'))
@pytest.mark.parametrize('dtype', ('float', 'double'))
def test_rng(target, rng, precision, dtype, t=124, offsets=(0, 0), keys=(0, 0), offset_values=None):
    if target == Target.GPU:
        pytest.importorskip('pycuda')
    if instruction_sets and {'neon', 'sve', 'vsx', 'rvv'}.intersection(instruction_sets) and rng == 'aesni':
        pytest.xfail('AES not yet implemented for this architecture')
    if rng == 'aesni' and len(keys) == 2:
        keys *= 2
    if offset_values is None:
        offset_values = offsets

    dh = ps.create_data_handling((2, 2), default_ghost_layers=0, default_target=target)
    f = dh.add_array("f", values_per_cell=4 if precision == 'float' else 2,
                     dtype=np.float32 if dtype == 'float' else np.float64)
    dh.fill(f.name, 42.0)

    rng_node = RNGs[(rng, precision)](dh.dim, offsets=offsets, keys=keys)
    assignments = [rng_node] + [SympyAssignment(f(i), s) for i, s in enumerate(rng_node.result_symbols)]
    kernel = ps.create_kernel(assignments, target=dh.default_target).compile()

    dh.all_to_gpu()
    kwargs = {'time_step': t}
    if offset_values != offsets:
        kwargs.update({k.name: v for k, v in zip(offsets, offset_values)})
    dh.run_kernel(kernel, **kwargs)
    dh.all_to_cpu()
    arr = dh.gather_array(f.name)
    assert np.logical_and(arr <= 1.0, arr >= 0).all()

    if rng == 'philox' and t == 124 and offsets == (0, 0) and keys == (0, 0) and dh.shape == (2, 2):
        int_reference = np.array([[[3576608082, 1252663339, 1987745383, 348040302],
                                   [1032407765, 970978240, 2217005168, 2424826293]],
                                  [[2958765206, 3725192638, 2623672781, 1373196132],
                                   [850605163, 1694561295, 3285694973, 2799652583]]])
    else:
        pytest.importorskip('randomgen')
        if rng == 'aesni':
            from randomgen import AESCounter
            int_reference = np.empty(dh.shape + (4,), dtype=int)
            for x in range(dh.shape[0]):
                for y in range(dh.shape[1]):
                    r = AESCounter(counter=t + (x + offset_values[0]) * 2 ** 32 + (y + offset_values[1]) * 2 ** 64,
                                   key=keys[0] + keys[1] * 2 ** 32 + keys[2] * 2 ** 64 + keys[3] * 2 ** 96,
                                   mode="sequence")
                    a, b = r.random_raw(size=2)
                    int_reference[x, y, :] = [a % 2 ** 32, a // 2 ** 32, b % 2 ** 32, b // 2 ** 32]
        else:
            from randomgen import Philox
            int_reference = np.empty(dh.shape + (4,), dtype=int)
            for x in range(dh.shape[0]):
                for y in range(dh.shape[1]):
                    r = Philox(counter=t + (x + offset_values[0]) * 2 ** 32 + (y + offset_values[1]) * 2 ** 64 - 1,
                               key=keys[0] + keys[1] * 2 ** 32, number=4, width=32, mode="sequence")
                    int_reference[x, y, :] = r.random_raw(size=4)

    if precision == 'float' or dtype == 'float':
        eps = np.finfo(np.float32).eps
    else:
        eps = np.finfo(np.float64).eps
    if rng == 'aesni':  # precision appears to be slightly worse
        eps = max(1e-12, 2 * eps)

    if precision == 'float':
        reference = int_reference * 2. ** -32 + 2. ** -33
    else:
        x = int_reference[:, :, 0::2]
        y = int_reference[:, :, 1::2]
        z = x ^ y << (53 - 32)
        reference = z * 2. ** -53 + 2. ** -54
    assert np.allclose(arr, reference, rtol=0, atol=eps)


@pytest.mark.parametrize('vectorized', (False, True))
@pytest.mark.parametrize('kind', ('value', 'symbol'))
def test_rng_offsets(kind, vectorized):
    if vectorized:
        test = test_rng_vectorized
        if not instruction_sets:
            pytest.skip("cannot detect CPU instruction set")
    else:
        test = test_rng
    if kind == 'value':
        test(instruction_sets[-1] if vectorized else Target.CPU, 'philox', 'float', 'float', t=8,
             offsets=(6, 7), keys=(5, 309))
    elif kind == 'symbol':
        offsets = (TypedSymbol("x0", np.uint32), TypedSymbol("y0", np.uint32))
        test(instruction_sets[-1] if vectorized else Target.GPU, 'philox', 'float', 'float', t=8,
             offsets=offsets, offset_values=(6, 7), keys=(5, 309))


@pytest.mark.parametrize('target', instruction_sets)
@pytest.mark.parametrize('rng', ('philox', 'aesni'))
@pytest.mark.parametrize('precision,dtype', (('float', 'float'), ('double', 'double')))
def test_rng_vectorized(target, rng, precision, dtype, t=130, offsets=(1, 3), keys=(0, 0), offset_values=None):
    if (target in ['neon', 'vsx', 'rvv'] or target.startswith('sve')) and rng == 'aesni':
        pytest.xfail('AES not yet implemented for this architecture')
    cpu_vectorize_info = {'assume_inner_stride_one': True, 'assume_aligned': True, 'instruction_set': target}

    dh = ps.create_data_handling((131, 131), default_ghost_layers=0, default_target=Target.CPU)
    f = dh.add_array("f", values_per_cell=4 if precision == 'float' else 2,
                     dtype=np.float32 if dtype == 'float' else np.float64, alignment=True)
    dh.fill(f.name, 42.0)
    ref = dh.add_array("ref", values_per_cell=4 if precision == 'float' else 2)

    rng_node = RNGs[(rng, precision)](dh.dim, offsets=offsets)
    assignments = [rng_node] + [SympyAssignment(ref(i), s) for i, s in enumerate(rng_node.result_symbols)]
    kernel = ps.create_kernel(assignments, target=dh.default_target).compile()

    kwargs = {'time_step': t}
    if offset_values is not None:
        kwargs.update({k.name: v for k, v in zip(offsets, offset_values)})
    dh.run_kernel(kernel, **kwargs)

    rng_node = RNGs[(rng, precision)](dh.dim, offsets=offsets)
    assignments = [rng_node] + [SympyAssignment(f(i), s) for i, s in enumerate(rng_node.result_symbols)]
    kernel = ps.create_kernel(assignments, target=dh.default_target, cpu_vectorize_info=cpu_vectorize_info).compile()

    dh.run_kernel(kernel, **kwargs)

    ref_data = dh.gather_array(ref.name)
    data = dh.gather_array(f.name)

    assert np.allclose(ref_data, data)


@pytest.mark.parametrize('vectorized', (False, True))
def test_rng_symbol(vectorized):
    """Make sure that the RNG symbol generator generates symbols and that the resulting code compiles"""
    cpu_vectorize_info = None
    if vectorized:
        if not instruction_sets:
            pytest.skip("cannot detect CPU instruction set")
        else:
            cpu_vectorize_info = {'assume_inner_stride_one': True, 'assume_aligned': True,
                                  'instruction_set': instruction_sets[-1]}

    dh = ps.create_data_handling((8, 8), default_ghost_layers=0, default_target=Target.CPU)
    f = dh.add_array("f", values_per_cell=2 * dh.dim, alignment=True)
    nc = NodeCollection([SympyAssignment(f(i), 0) for i in range(f.shape[-1])])
    subexpressions = []
    rng_symbol_gen = random_symbol(subexpressions, dim=dh.dim)
    for i in range(f.shape[-1]):
        nc.all_assignments[i] = SympyAssignment(nc.all_assignments[i].lhs, next(rng_symbol_gen))
    symbols = [a.rhs for a in nc.all_assignments]
    [nc.all_assignments.insert(0, subexpression) for subexpression in subexpressions]
    assert len(symbols) == f.shape[-1] and len(set(symbols)) == f.shape[-1]
    ps.create_kernel(nc, target=dh.default_target, cpu_vectorize_info=cpu_vectorize_info).compile()


@pytest.mark.parametrize('vectorized', (False, True))
def test_staggered(vectorized):
    """Make sure that the RNG counter can be substituted during loop cutting"""

    dh = ps.create_data_handling((8, 8), default_ghost_layers=0, default_target=Target.CPU)
    j = dh.add_array("j", values_per_cell=dh.dim, field_type=ps.FieldType.STAGGERED_FLUX)
    a = ps.AssignmentCollection([ps.Assignment(j.staggered_access(n), 0) for n in j.staggered_stencil])
    rng_symbol_gen = random_symbol(a.subexpressions, dim=dh.dim, rng_node=PhiloxTwoDoubles)
    a.main_assignments[0] = ps.Assignment(a.main_assignments[0].lhs, next(rng_symbol_gen))
    kernel = ps.create_staggered_kernel(a, target=dh.default_target).compile()

    if not vectorized:
        return
    if not instruction_sets:
        pytest.skip("cannot detect CPU instruction set")
    pytest.importorskip('islpy')
    cpu_vectorize_info = {'assume_inner_stride_one': True, 'assume_aligned': False,
                          'instruction_set': instruction_sets[-1]}

    dh.fill(j.name, 867)
    dh.run_kernel(kernel, seed=5, time_step=309)
    ref_data = dh.gather_array(j.name)

    kernel2 = ps.create_staggered_kernel(a, target=dh.default_target, cpu_vectorize_info=cpu_vectorize_info).compile()

    dh.fill(j.name, 867)
    dh.run_kernel(kernel2, seed=5, time_step=309)
    data = dh.gather_array(j.name)

    assert np.allclose(ref_data, data)
