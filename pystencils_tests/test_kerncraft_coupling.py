import os

import numpy as np
import pytest
import sympy as sp
import kerncraft

from pystencils import Assignment, Field
from pystencils.cpu import create_kernel
from pystencils.kerncraft_coupling import KerncraftParameters, PyStencilsKerncraftKernel
from pystencils.kerncraft_coupling.generate_benchmark import generate_benchmark

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "kerncraft_inputs")


@pytest.mark.kerncraft
def test_compilation():
    machine_file_path = os.path.join(INPUT_FOLDER, "Example_SandyBridgeEP_E5-2680.yml")
    machine = kerncraft.machinemodel.MachineModel(path_to_yaml=machine_file_path)

    kernel_file_path = os.path.join(INPUT_FOLDER, "2d-5pt.c")
    with open(kernel_file_path) as kernel_file:
        reference_kernel = kerncraft.kernel.KernelCode(kernel_file.read(), machine=machine, filename=kernel_file_path)
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
def analysis(kernel, model='ecmdata'):
    machine_file_path = os.path.join(INPUT_FOLDER, "Example_SandyBridgeEP_E5-2680.yml")
    machine = kerncraft.machinemodel.MachineModel(path_to_yaml=machine_file_path)
    if model == 'ecmdata':
        model = kerncraft.models.ECMData(kernel, machine, KerncraftParameters())
    elif model == 'ecm':
        model = kerncraft.models.ECM(kernel, machine, KerncraftParameters())
        # model.analyze()
        # model.plot()
    elif model == 'benchmark':
        model = kerncraft.models.Benchmark(kernel, machine, KerncraftParameters())
    else:
        model = kerncraft.models.ECM(kernel, machine, KerncraftParameters())
    model.analyze()
    return model


@pytest.mark.kerncraft
def test_3d_7pt_OSACA():
    # Make sure you use the intel compiler
    size = [20, 200, 200]
    kernel_file_path = os.path.join(INPUT_FOLDER, "3d-7pt.c")
    machine_file_path = os.path.join(INPUT_FOLDER, "Example_SandyBridgeEP_E5-2680.yml")
    machine = kerncraft.machinemodel.MachineModel(path_to_yaml=machine_file_path)
    with open(kernel_file_path) as kernel_file:
        reference_kernel = kerncraft.kernel.KernelCode(kernel_file.read(), machine=machine, filename=kernel_file_path)
    reference_kernel.set_constant('M', size[0])
    reference_kernel.set_constant('N', size[1])
    assert size[1] == size[2]
    analysis(reference_kernel, model='ecm')

    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=0)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=0)
    s = sp.Symbol("s")
    rhs = a[0, -1, 0] + a[0, 1, 0] + a[-1, 0, 0] + a[1, 0, 0] + a[0, 0, -1] + a[0, 0, 1]

    update_rule = Assignment(b[0, 0, 0], s * rhs)
    ast = create_kernel([update_rule])
    k = PyStencilsKerncraftKernel(ast, machine)
    analysis(k, model='ecm')
    assert reference_kernel._flops == k._flops
    # assert reference.results['cl throughput'] == analysis.results['cl throughput']


@pytest.mark.kerncraft
def test_2d_5pt():
    size = [30, 50, 3]
    kernel_file_path = os.path.join(INPUT_FOLDER, "2d-5pt.c")
    with open(kernel_file_path) as kernel_file:
        reference_kernel = kerncraft.kernel.KernelCode(kernel_file.read(), machine=None, filename=kernel_file_path)
    reference = analysis(reference_kernel)

    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=1)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=1)
    s = sp.Symbol("s")
    rhs = a[0, -1](0) + a[0, 1] + a[-1, 0] + a[1, 0]
    update_rule = Assignment(b[0, 0], s * rhs)
    ast = create_kernel([update_rule])
    k = PyStencilsKerncraftKernel(ast)
    result = analysis(k)

    for e1, e2 in zip(reference.results['cycles'], result.results['cycles']):
        assert e1 == e2


@pytest.mark.kerncraft
def test_3d_7pt():
    size = [30, 50, 50]
    kernel_file_path = os.path.join(INPUT_FOLDER, "3d-7pt.c")
    with open(kernel_file_path) as kernel_file:
        reference_kernel = kerncraft.kernel.KernelCode(kernel_file.read(), machine=None, filename=kernel_file_path)
    reference_kernel.set_constant('M', size[0])
    reference_kernel.set_constant('N', size[1])
    assert size[1] == size[2]
    reference = analysis(reference_kernel)

    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=0)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=0)
    s = sp.Symbol("s")
    rhs = a[0, -1, 0] + a[0, 1, 0] + a[-1, 0, 0] + a[1, 0, 0] + a[0, 0, -1] + a[0, 0, 1]

    update_rule = Assignment(b[0, 0, 0], s * rhs)
    ast = create_kernel([update_rule])
    k = PyStencilsKerncraftKernel(ast)
    result = analysis(k)

    for e1, e2 in zip(reference.results['cycles'], result.results['cycles']):
        assert e1 == e2
