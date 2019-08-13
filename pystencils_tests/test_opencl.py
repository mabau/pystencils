import sympy as sp

import pystencils
from pystencils.backends.cuda_backend import CudaBackend
from pystencils.backends.opencl_backend import OpenClBackend


def test_opencl_backend():
    z, y, x = pystencils.fields("z, y, x: [2d]")

    assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sp.log(x[0, 0] * y[0, 0])
    })

    print(assignments)

    ast = pystencils.create_kernel(assignments, target='gpu')

    print(ast)

    code = pystencils.show_code(ast, custom_backend=CudaBackend())
    print(code)

    opencl_code = pystencils.show_code(ast, custom_backend=OpenClBackend())
    print(opencl_code)


if __name__ == '__main__':
    test_opencl_backend()
