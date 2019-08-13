import pystencils.data_types
from pystencils.astnodes import Node
from pystencils.backends.cbackend import CustomSympyPrinter, generate_c
from pystencils.backends.cuda_backend import CudaBackend, CudaSympyPrinter


def generate_opencl(astnode: Node, signature_only: bool = False) -> str:
    """Prints an abstract syntax tree node as CUDA code.

    Args:
        astnode: KernelFunction node to generate code for
        signature_only: if True only the signature is printed

    Returns:
        C-like code for the ast node and its descendants
    """
    return generate_c(astnode, signature_only, dialect='opencl')


class OpenClBackend(CudaBackend):

    def __init__(self,
                 sympy_printer=None,
                 signature_only=False):
        if not sympy_printer:
            sympy_printer = OpenClSympyPrinter()

        super().__init__(sympy_printer, signature_only)
        self._dialect = 'opencl'

    def _print_Type(self, node):
        code = super()._print_Type(node)
        if isinstance(node, pystencils.data_types.PointerType):
            return "__global " + code
        else:
            return code

    def _print_ThreadBlockSynchronization(self, node):
        raise NotImplementedError()

    def _print_TextureDeclaration(self, node):
        raise NotImplementedError()


class OpenClSympyPrinter(CudaSympyPrinter):
    language = "OpenCL"

    DIMENSION_MAPPING = {
        'x': '0',
        'y': '1',
        'z': '2'
    }
    INDEXING_FUNCTION_MAPPING = {
        'blockIdx': 'get_group_id',
        'threadIdx': 'get_local_id',
        'blockDim': 'get_local_size',
        'gridDim': 'get_global_size'
    }

    def _print_ThreadIndexingSymbol(self, node):
        symbol_name: str = node.name
        function_name, dimension = tuple(symbol_name.split("."))
        dimension = self.DIMENSION_MAPPING[dimension]
        function_name = self.INDEXING_FUNCTION_MAPPING[function_name]
        return f"{function_name}({dimension})"

    def _print_TextureAccess(self, node):
        raise NotImplementedError()

    # Avoid usage of CUDA intrinsics
    _print_Function = CustomSympyPrinter._print_Function
