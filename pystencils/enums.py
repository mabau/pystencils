from enum import Enum, auto


class Target(Enum):
    CPU = auto()
    GPU = auto()
    OPENCL = auto()


class Backend(Enum):
    C = auto()
    LLVM = auto()
    CUDA = auto()
    OPENCL = auto()
