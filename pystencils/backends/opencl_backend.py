from pystencils.backends.cuda_backend import CudaBackend
from pystencils.backends.cbackend import generate_c
from pystencils.astnodes import Node

def generate_opencl(astnode: Node, signature_only: bool = False) -> str:
    """Prints an abstract syntax tree node as CUDA code.

    Args:
        astnode: KernelFunction node to generate code for
        signature_only: if True only the signature is printed

    Returns:
        C-like code for the ast node and its descendants
    """
    return generate_c(astnode, signature_only, dialect='opencl')


class OpenCLBackend(CudaBackend):
    pass