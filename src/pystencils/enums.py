from enum import Flag, auto


class Target(Flag):
    """
    The Target enumeration represents all possible targets that can be used for the code generation.
    """

    #   ------------------ Component Flags - Do Not Use Directly! -------------------------------------------

    _CPU = auto()

    _VECTOR = auto()

    _X86 = auto()
    _SSE = auto()
    _AVX = auto()
    _AVX512 = auto()

    _ARM = auto()
    _NEON = auto()
    _SVE = auto()

    _GPU = auto()

    _CUDA = auto()

    _AUTOMATIC = auto()

    #   ------------------ Actual Targets -------------------------------------------------------------------

    CurrentCPU = _CPU | _AUTOMATIC
    """
    Auto-best CPU target. 
    
    `CurrentCPU` causes the code generator to automatically select a CPU target according to CPUs found
    on the current machine and runtime environment.
    """

    GenericCPU = _CPU
    """Generic CPU target.
    
    Generate the kernel for a generic multicore CPU architecture. This opens up all architecture-independent
    optimizations including OpenMP, but no vectorization.
    """

    CPU = GenericCPU
    """Alias for backward-compatibility"""

    X86_SSE = _CPU | _VECTOR | _X86 | _SSE
    """x86 architecture with SSE vector extensions."""

    X86_AVX = _CPU | _VECTOR | _X86 | _AVX
    """x86 architecture with AVX vector extensions."""

    X86_AVX512 = _CPU | _VECTOR | _X86 | _AVX512
    """x86 architecture with AVX512 vector extensions."""

    ARM_NEON = _CPU | _VECTOR | _ARM | _NEON
    """ARM architecture with NEON vector extensions"""

    ARM_SVE = _CPU | _VECTOR | _ARM | _SVE
    """ARM architecture with SVE vector extensions"""

    CurrentGPU = _GPU | _AUTOMATIC
    """
    Auto-best GPU target.

    `CurrentGPU` causes the code generator to automatically select a GPU target according to GPU devices
    found on the current machine and runtime environment.
    """

    GenericCUDA = _GPU | _CUDA
    """
    Generic CUDA GPU target.

    Generate a CUDA kernel for a generic Nvidia GPU.
    """

    GPU = GenericCUDA
    """Alias for backward compatibility."""

    def is_automatic(self) -> bool:
        return Target._AUTOMATIC in self

    def is_cpu(self) -> bool:
        return Target._CPU in self

    def is_vector_cpu(self) -> bool:
        return self.is_cpu() and Target._VECTOR in self

    def is_gpu(self) -> bool:
        return Target._GPU in self
