from enum import Enum, auto


class Target(Enum):
    """
    The Target enumeration represents all possible targets that can be used for the code generation.
    """
    CPU = auto()
    """
    Target CPU architecture.
    """
    GPU = auto()
    """
    Target GPU architecture.
    """
    OPENCL = auto()
    """
    Target all architectures OpenCL covers (Thus both, Target and Backend)
    """


class Backend(Enum):
    """
    The Backend enumeration represents all possible backends that can be used for the code generation.
    Backends and targets must be combined with care. For example CPU as a target and CUDA as a backend makes no sense.
    """
    C = auto()
    """
    Use the C Backend of pystencils.
    """
    LLVM = auto()
    r"""
    Use the ``llvmlite`` package to transform the pystensilc AST to the LLVM ir. 
    From this point all of LLVMs optimisations can be used.
    """
    CUDA = auto()
    """
    Use the CUDA backend to generate code for NVIDIA GPUs.
    """
    OPENCL = auto()
    """
    Use the OpenCL backend to generate code for OpenCL.
    """
