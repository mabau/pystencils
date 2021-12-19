from subprocess import CalledProcessError

import pytest

import pystencils
import pystencils.cpu.cpujit
from pystencils.backends.cbackend import CBackend
from pystencils.backends.cuda_backend import CudaBackend
from pystencils.enums import Target


class ScreamingBackend(CBackend):

    def _print(self, node):
        normal_code = super()._print(node)
        return normal_code.upper()


class ScreamingGpuBackend(CudaBackend):

    def _print(self, node):
        normal_code = super()._print(node)
        return normal_code.upper()


def test_custom_backends_cpu():
    z, y, x = pystencils.fields("z, y, x: [2d]")

    normal_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * x[0, 0] * y[0, 0])], [])

    ast = pystencils.create_kernel(normal_assignments, target=Target.CPU)
    pystencils.show_code(ast, ScreamingBackend())
    with pytest.raises(CalledProcessError):
        pystencils.cpu.cpujit.make_python_function(ast, custom_backend=ScreamingBackend())


def test_custom_backends_gpu():
    pytest.importorskip('pycuda')
    import pycuda.driver
    import pystencils.gpucuda.cudajit

    z, x, y = pystencils.fields("z, y, x: [2d]")

    normal_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * x[0, 0] * y[0, 0])], [])

    ast = pystencils.create_kernel(normal_assignments, target=Target.GPU)
    pystencils.show_code(ast, ScreamingGpuBackend())
    with pytest.raises(pycuda.driver.CompileError):
        pystencils.gpucuda.cudajit.make_python_function(ast, custom_backend=ScreamingGpuBackend())
