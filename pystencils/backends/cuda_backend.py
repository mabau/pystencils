from os.path import dirname, join

from pystencils.astnodes import Node
from pystencils.backends.cbackend import CBackend, CustomSympyPrinter, generate_c
from pystencils.fast_approximation import fast_division, fast_inv_sqrt, fast_sqrt
from pystencils.interpolation_astnodes import DiffInterpolatorAccess, InterpolationMode

with open(join(dirname(__file__), 'cuda_known_functions.txt')) as f:
    lines = f.readlines()
    CUDA_KNOWN_FUNCTIONS = {l.strip(): l.strip() for l in lines if l}


def generate_cuda(astnode: Node, signature_only: bool = False) -> str:
    """Prints an abstract syntax tree node as CUDA code.

    Args:
        astnode: KernelFunction node to generate code for
        signature_only: if True only the signature is printed

    Returns:
        C-like code for the ast node and its descendants
    """
    return generate_c(astnode, signature_only, dialect='cuda')


class CudaBackend(CBackend):

    def __init__(self, sympy_printer=None,
                 signature_only=False):
        if not sympy_printer:
            sympy_printer = CudaSympyPrinter()

        super().__init__(sympy_printer, signature_only, dialect='cuda')

    def _print_SharedMemoryAllocation(self, node):
        code = "__shared__ {dtype} {name}[{num_elements}];"
        return code.format(dtype=node.symbol.dtype,
                           name=self.sympy_printer.doprint(node.symbol.name),
                           num_elements='*'.join([str(s) for s in node.shared_mem.shape]))

    @staticmethod
    def _print_ThreadBlockSynchronization(node):
        code = "__synchtreads();"
        return code

    def _print_TextureDeclaration(self, node):

        if node.texture.field.dtype.numpy_dtype.itemsize > 4:
            code = "texture<fp_tex_%s, cudaTextureType%iD, cudaReadModeElementType> %s;" % (
                str(node.texture.field.dtype),
                node.texture.field.spatial_dimensions,
                node.texture
            )
        else:
            code = "texture<%s, cudaTextureType%iD, cudaReadModeElementType> %s;" % (
                str(node.texture.field.dtype),
                node.texture.field.spatial_dimensions,
                node.texture
            )
        return code

    def _print_SkipIteration(self, _):
        return "return;"


class CudaSympyPrinter(CustomSympyPrinter):
    language = "CUDA"

    def __init__(self):
        super(CudaSympyPrinter, self).__init__()
        self.known_functions.update(CUDA_KNOWN_FUNCTIONS)

    def _print_TextureAccess(self, node):
        dtype = node.texture.field.dtype.numpy_dtype

        if type(node) == DiffInterpolatorAccess:
            # cubicTex3D_1st_derivative_x(texture tex, float3 coord)
            template = f"cubicTex%iD_1st_{'xyz'[node.diff_coordinate_idx]}(%s, %s)"
        elif node.texture.interpolation_mode == InterpolationMode.CUBIC_SPLINE:
            template = "cubicTex%iDSimple(%s, %s)"
        else:
            if dtype.itemsize > 4:
                # Use PyCuda hack!
                # https://github.com/inducer/pycuda/blob/master/pycuda/cuda/pycuda-helpers.hpp
                template = "fp_tex%iD(%s, %s)"
            else:
                template = "tex%iD(%s, %s)"

        code = template % (
            node.texture.field.spatial_dimensions,
            str(node.texture),
            # + 0.5 comes from Nvidia's staggered indexing
            ', '.join(self._print(o + 0.5) for o in reversed(node.offsets))
        )
        return code

    def _print_Function(self, expr):
        if isinstance(expr, fast_division):
            return "__fdividef(%s, %s)" % tuple(self._print(a) for a in expr.args)
        elif isinstance(expr, fast_sqrt):
            return "__fsqrt_rn(%s)" % tuple(self._print(a) for a in expr.args)
        elif isinstance(expr, fast_inv_sqrt):
            return "__frsqrt_rn(%s)" % tuple(self._print(a) for a in expr.args)
        return super()._print_Function(expr)
