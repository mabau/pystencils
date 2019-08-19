from subprocess import CalledProcessError

import pycuda.driver
import pytest
import sympy

import pystencils
import pystencils.cpu.cpujit
import pystencils.gpucuda.cudajit
from pystencils.backends.cbackend import CBackend
from pystencils.backends.cuda_backend import CudaBackend


class ScreamingBackend(CBackend):

    def _print(self, node):
        normal_code = super()._print(node)
        return normal_code.upper()


class ScreamingGpuBackend(CudaBackend):

    def _print(self, node):
        normal_code = super()._print(node)
        return normal_code.upper()


def test_custom_backends():
    z, x, y = pystencils.fields("z, y, x: [2d]")

    normal_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * sympy.log(x[0, 0] * y[0, 0]))], [])

    ast = pystencils.create_kernel(normal_assignments, target='cpu')
    print(pystencils.show_code(ast, ScreamingBackend()))
    with pytest.raises(CalledProcessError):
        pystencils.cpu.cpujit.make_python_function(ast, custom_backend=ScreamingBackend())

    ast = pystencils.create_kernel(normal_assignments, target='gpu')
    print(pystencils.show_code(ast, ScreamingGpuBackend()))
    with pytest.raises(pycuda.driver.CompileError):
        pystencils.gpucuda.cudajit.make_python_function(ast, custom_backend=ScreamingGpuBackend())
