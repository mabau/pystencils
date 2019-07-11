import numpy as np
import sympy as sp

from pystencils import Assignment, Field, create_kernel


def meassure():
    size = [30, 50, 3]
    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=1)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=1)
    s = sp.Symbol("s")
    rhs = a[0, -1](0) + a[0, 1] + a[-1, 0] + a[1, 0]
    updateRule = Assignment(b[0, 0], s * rhs)
    print(updateRule)

    ast = create_kernel([updateRule])

    # benchmark = generate_benchmark(ast)
    # main = benchmark[0]
    # kernel = benchmark[1]
    # with open('src/main.cpp', 'w') as file:
    #     file.write(main)
    # with open('src/kernel.cpp', 'w') as file:
    #     file.write(kernel)

    func = ast.compile({'omega': 2/3})

    from pystencils.kerncraft_coupling.generate_benchmark import generate_benchmark
    from pystencils.kerncraft_coupling import BenchmarkAnalysis
    from pystencils.kerncraft_coupling.kerncraft_interface import PyStencilsKerncraftKernel, KerncraftParameters
    from kerncraft.machinemodel import MachineModel
    from kerncraft.models import ECMData


    machineFilePath = "../pystencils_tests/kerncraft_inputs/default_machine_file.yaml"
    machine = MachineModel(path_to_yaml=machineFilePath)


    benchmark = BenchmarkAnalysis(ast, machine)
    #TODO what do i want to do with benchmark?

    kernel = PyStencilsKerncraftKernel(ast)
    model = ECMData(kernel, machine, KerncraftParameters())
    model.analyze()
    model.report()


if __name__ == "__main__":
    meassure()
