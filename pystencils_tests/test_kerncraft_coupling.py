import numpy as np
import pytest
import sympy as sp
from pathlib import Path

from kerncraft.kernel import KernelCode
from kerncraft.machinemodel import MachineModel
from kerncraft.models import ECM, ECMData, Benchmark

from pystencils import Assignment, Field, fields
from pystencils.cpu import create_kernel
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


@pytest.mark.kerncraft
def analysis(kernel, machine, model='ecmdata'):
    if model == 'ecmdata':
        model = ECMData(kernel, machine, KerncraftParameters())
    elif model == 'ecm':
        model = ECM(kernel, machine, KerncraftParameters())
        # model.analyze()
        # model.plot()
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
    k = PyStencilsKerncraftKernel(ast, machine=machine_model)
    analysis(k, machine_model, model='ecm')
    assert reference_kernel._flops == k._flops
    # assert reference.results['cl throughput'] == analysis.results['cl throughput']


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
