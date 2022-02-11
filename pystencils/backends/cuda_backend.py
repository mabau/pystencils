from pystencils.astnodes import Node
from pystencils.backends.cbackend import CBackend, CustomSympyPrinter, generate_c
from pystencils.enums import Backend
from pystencils.fast_approximation import fast_division, fast_inv_sqrt, fast_sqrt


def generate_cuda(ast_node: Node, signature_only: bool = False, custom_backend=None, with_globals=True) -> str:
    """Prints an abstract syntax tree node as CUDA code.

    Args:
        ast_node: ast representation of kernel
        signature_only: generate signature without function body
        custom_backend: use own custom printer for code generation
        with_globals: enable usage of global variables

    Returns:
        CUDA code for the ast node and its descendants
    """
    return generate_c(ast_node, signature_only, dialect=Backend.CUDA,
                      custom_backend=custom_backend, with_globals=with_globals)


class CudaBackend(CBackend):

    def __init__(self, sympy_printer=None,
                 signature_only=False):
        if not sympy_printer:
            sympy_printer = CudaSympyPrinter()

        super().__init__(sympy_printer, signature_only, dialect=Backend.CUDA)

    def _print_SharedMemoryAllocation(self, node):
        dtype = node.symbol.dtype
        name = self.sympy_printer.doprint(node.symbol.name)
        num_elements = '*'.join([str(s) for s in node.shared_mem.shape])
        code = f"__shared__ {dtype} {name}[{num_elements}];"
        return code

    @staticmethod
    def _print_ThreadBlockSynchronization(_):
        return "__synchtreads();"

    def _print_TextureDeclaration(self, node):
        cond = node.texture.field.dtype.numpy_dtype.itemsize > 4
        return f'texture<{"fp_tex_" if cond else ""}{str(node.texture.field.dtype)}, ' \
               f'cudaTextureType{node.texture.field.spacial_dimensions}D, cudaReadModeElementType> {node.texture};'

    def _print_SkipIteration(self, _):
        return "return;"


class CudaSympyPrinter(CustomSympyPrinter):
    language = "CUDA"

    def __init__(self):
        super(CudaSympyPrinter, self).__init__()

    def _print_Function(self, expr):
        if isinstance(expr, fast_division):
            assert len(expr.args) == 2, f"__fdividef has two arguments, but {len(expr.args)} where given"
            return f"__fdividef({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        elif isinstance(expr, fast_sqrt):
            assert len(expr.args) == 1, f"__fsqrt_rn has one argument, but {len(expr.args)} where given"
            return f"__fsqrt_rn({self._print(expr.args[0])})"
        elif isinstance(expr, fast_inv_sqrt):
            assert len(expr.args) == 1, f"__frsqrt_rn has one argument, but {len(expr.args)} where given"
            return f"__frsqrt_rn({self._print(expr.args[0])})"
        return super()._print_Function(expr)
