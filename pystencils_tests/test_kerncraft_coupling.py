import numpy as np
import pytest
import sympy as sp
from pathlib import Path

from kerncraft.kernel import KernelCode
from kerncraft.machinemodel import MachineModel
from kerncraft.models import ECM, ECMData, Benchmark

import pystencils as ps
from pystencils import Assignment, Field
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets, get_vector_instruction_set
from pystencils.cpu import create_kernel
from pystencils.datahandling import create_data_handling
from pystencils.kerncraft_coupling import KerncraftParameters, PyStencilsKerncraftKernel
from pystencils.kerncraft_coupling.generate_benchmark import generate_benchmark, run_c_benchmark
from pystencils.timeloop import TimeLoop

SCRIPT_FOLDER = Path(__file__).parent
INPUT_FOLDER = SCRIPT_FOLDER / "kerncraft_inputs"


@pytest.mark.kerncraft
def test_compilation():
    machine_file_path = INPUT_FOLDER / "Example_SandyBridgeEP_E5-2680.yml"
    machine = MachineModel(path_to_yaml=machine_file_path)

    kernel_file_path = INPUT_FOLDER / "2d-5pt.c"
    with open(kernel_file_path) as kernel_file:
        reference_kernel = KernelCode(kernel_file.read(), machine=machine, filename=kernel_file_path)
        reference_kernel.get_kernel_header(name='test_kernel')
        reference_kernel.get_kernel_code(name='test_kernel')
        reference_kernel.get_main_code(kernel_function_name='test_kernel')

    size = [30, 50, 3]
    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=1)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=1)
    s = sp.Symbol("s")
    rhs = a[0, -1](0) + a[0, 1] + a[-1, 0] + a[1, 0]
    update_rule = Assignment(b[0, 0], s * rhs)
    ast = create_kernel([update_rule])
    mine = generate_benchmark(ast, likwid=False)
    print(mine)


def analysis(kernel, machine, model='ecmdata'):
    if model == 'ecmdata':
        model = ECMData(kernel, machine, KerncraftParameters())
    elif model == 'ecm':
        model = ECM(kernel, machine, KerncraftParameters())
    elif model == 'benchmark':
        model = Benchmark(kernel, machine, KerncraftParameters())
    else:
        model = ECM(kernel, machine, KerncraftParameters())
    model.analyze()
    return model


@pytest.mark.kerncraft
def test_3d_7pt_osaca():

    size = [20, 200, 200]
    kernel_file_path = INPUT_FOLDER / "3d-7pt.c"
    machine_file_path = INPUT_FOLDER / "Example_SandyBridgeEP_E5-2680.yml"
    machine_model = MachineModel(path_to_yaml=machine_file_path)
    with open(kernel_file_path) as kernel_file:
        reference_kernel = KernelCode(kernel_file.read(), machine=machine_model, filename=kernel_file_path)
    reference_kernel.set_constant('M', size[0])
    reference_kernel.set_constant('N', size[1])
    assert size[1] == size[2]
    analysis(reference_kernel, machine_model, model='ecm')

    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=0)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=0)
    s = sp.Symbol("s")
    rhs = a[0, -1, 0] + a[0, 1, 0] + a[-1, 0, 0] + a[1, 0, 0] + a[0, 0, -1] + a[0, 0, 1]

    update_rule = Assignment(b[0, 0, 0], s * rhs)
    ast = create_kernel([update_rule])
    k = PyStencilsKerncraftKernel(ast, machine=machine_model, debug_print=True)
    analysis(k, machine_model, model='ecm')
    assert reference_kernel._flops == k._flops

    path, lock = k.get_kernel_code(openmp=True)
    with open(path) as kernel_file:
        assert "#pragma omp parallel" in kernel_file.read()

    path, lock = k.get_main_code()
    with open(path) as kernel_file:
        assert "likwid_markerInit();" in kernel_file.read()


@pytest.mark.kerncraft
def test_2d_5pt():
    machine_file_path = INPUT_FOLDER / "Example_SandyBridgeEP_E5-2680.yml"
    machine = MachineModel(path_to_yaml=machine_file_path)

    size = [30, 50, 3]
    kernel_file_path = INPUT_FOLDER / "2d-5pt.c"
    with open(kernel_file_path) as kernel_file:
        reference_kernel = KernelCode(kernel_file.read(), machine=machine, 
                                      filename=kernel_file_path)
    reference = analysis(reference_kernel, machine)

    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=1)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=1)
    s = sp.Symbol("s")
    rhs = a[0, -1](0) + a[0, 1] + a[-1, 0] + a[1, 0]
    update_rule = Assignment(b[0, 0], s * rhs)
    ast = create_kernel([update_rule])
    k = PyStencilsKerncraftKernel(ast, machine)
    result = analysis(k, machine)

    for e1, e2 in zip(reference.results['cycles'], result.results['cycles']):
        assert e1 == e2


@pytest.mark.kerncraft
def test_3d_7pt():
    machine_file_path = INPUT_FOLDER / "Example_SandyBridgeEP_E5-2680.yml"
    machine = MachineModel(path_to_yaml=machine_file_path)

    size = [30, 50, 50]
    kernel_file_path = INPUT_FOLDER / "3d-7pt.c"
    with open(kernel_file_path) as kernel_file:
        reference_kernel = KernelCode(kernel_file.read(), machine=machine,
                                      filename=kernel_file_path)
    reference_kernel.set_constant('M', size[0])
    reference_kernel.set_constant('N', size[1])
    assert size[1] == size[2]
    reference = analysis(reference_kernel, machine)

    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=0)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=0)
    s = sp.Symbol("s")
    rhs = a[0, -1, 0] + a[0, 1, 0] + a[-1, 0, 0] + a[1, 0, 0] + a[0, 0, -1] + a[0, 0, 1]

    update_rule = Assignment(b[0, 0, 0], s * rhs)
    ast = create_kernel([update_rule])
    k = PyStencilsKerncraftKernel(ast, machine)
    result = analysis(k, machine)

    for e1, e2 in zip(reference.results['cycles'], result.results['cycles']):
        assert e1 == e2


@pytest.mark.kerncraft
def test_benchmark():
    size = [30, 50, 50]
    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=0)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=0)
    s = sp.Symbol("s")
    rhs = a[0, -1, 0] + a[0, 1, 0] + a[-1, 0, 0] + a[1, 0, 0] + a[0, 0, -1] + a[0, 0, 1]

    update_rule = Assignment(b[0, 0, 0], s * rhs)
    ast = create_kernel([update_rule])

    c_benchmark_run = run_c_benchmark(ast, inner_iterations=1000, outer_iterations=1)

    kernel = ast.compile()
    a = np.full(size, fill_value=0.23)
    b = np.full(size, fill_value=0.23)

    timeloop = TimeLoop(steps=1)
    timeloop.add_call(kernel, {'a': a, 'b': b, 's': 0.23})

    timeloop_time = timeloop.benchmark(number_of_time_steps_for_estimation=1)

    np.testing.assert_almost_equal(c_benchmark_run, timeloop_time, decimal=4)


@pytest.mark.kerncraft
def test_benchmark_vectorized():
    instruction_sets = get_supported_instruction_sets()
    if not instruction_sets:
        pytest.skip("cannot detect CPU instruction set")

    for vec in instruction_sets:
        dh = create_data_handling((20, 20, 20), periodicity=True)

        width = get_vector_instruction_set(instruction_set=vec)['width'] * 8

        a = dh.add_array("a", values_per_cell=1, alignment=width)
        b = dh.add_array("b", values_per_cell=1, alignment=width)

        rhs = a[0, -1, 0] + a[0, 1, 0] + a[-1, 0, 0] + a[1, 0, 0] + a[0, 0, -1] + a[0, 0, 1]
        update_rule = Assignment(b[0, 0, 0], rhs)

        opt = {'instruction_set': vec, 'assume_aligned': True, 'nontemporal': True, 'assume_inner_stride_one': True}
        ast = ps.create_kernel(update_rule, cpu_vectorize_info=opt)

        run_c_benchmark(ast, 5)
