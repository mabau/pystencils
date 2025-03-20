from __future__ import annotations

from enum import Flag, auto
from warnings import warn
from functools import cache


class Target(Flag):
    """
    The Target enumeration represents all possible targets that can be used for code generation.
    """

    #   ------------------ Component Flags - Do Not Use Directly! -------------------------------------------

    _CPU = auto()

    _VECTOR = auto()

    _X86 = auto()
    _SSE = auto()
    _AVX = auto()
    _AVX512 = auto()
    _VL = auto()
    _FP16 = auto()

    _ARM = auto()
    _NEON = auto()
    _SVE = auto()

    _GPU = auto()

    _CUDA = auto()
    _HIP = auto()

    _SYCL = auto()

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

    X86_AVX512_FP16 = _CPU | _VECTOR | _X86 | _AVX512 | _FP16
    """x86 architecture with AVX512 vector extensions and fp16-support."""

    ARM_NEON = _CPU | _VECTOR | _ARM | _NEON
    """ARM architecture with NEON vector extensions"""

    ARM_SVE = _CPU | _VECTOR | _ARM | _SVE
    """ARM architecture with SVE vector extensions"""

    CurrentGPU = _GPU | _AUTOMATIC
    """Auto-best GPU target.

    `CurrentGPU` causes the code generator to automatically select a GPU target according to GPU devices
    found on the current machine and runtime environment.
    """

    CUDA = _GPU | _CUDA
    """Generic CUDA GPU target.

    Generate a CUDA kernel for a generic Nvidia GPU.
    """

    HIP = _GPU | _HIP
    """Generic HIP GPU target.
    
    Generate a HIP kernel for generic AMD or NVidia GPUs.
    """

    GPU = CurrentGPU
    """Alias for `Target.CurrentGPU`, for backward compatibility."""

    SYCL = _SYCL
    """SYCL kernel target.
    
    Generate a function to be called within a SYCL parallel command.

    ..  note::
        The SYCL target is experimental and not thoroughly tested yet.
    """

    def is_automatic(self) -> bool:
        """Determine if this target is a proxy target that is automatically resolved
        according to the runtime environment."""
        return Target._AUTOMATIC in self

    def is_cpu(self) -> bool:
        """Determine if this target is a CPU target."""
        return Target._CPU in self

    def is_vector_cpu(self) -> bool:
        """Determine if this target is a vector CPU target."""
        return self.is_cpu() and Target._VECTOR in self

    def is_gpu(self) -> bool:
        """Determine if this target is a GPU target.
        
        This refers to targets for the CUDA and HIP family of platforms.
        `Target.SYCL` is *not* a GPU target.
        """
        return Target._GPU in self

    @staticmethod
    def auto_cpu() -> Target:
        """Return the most capable vector CPU target available on the current machine."""
        avail_targets = _available_vector_targets()
        if avail_targets:
            return avail_targets.pop()
        else:
            return Target.GenericCPU
        
    @staticmethod
    def auto_gpu() -> Target:
        """Return the GPU target available in the current runtime environment.
        
        Raises:
            RuntimeError: If `cupy` is not installed and therefore no GPU runtime is available.
        """
        try:
            import cupy

            if cupy.cuda.runtime.is_hip:
                return Target.HIP
            else:
                return Target.CUDA
        except ImportError:
            raise RuntimeError("Cannot infer GPU target since cupy is not installed.")
        
    @staticmethod
    def available_targets() -> list[Target]:
        """List available"""
        targets = [Target.GenericCPU]
        try:
            import cupy  # noqa: F401
            targets.append(Target.auto_gpu())
        except ImportError:
            pass

        targets += Target.available_vector_cpu_targets()
        return targets

    @staticmethod
    def available_vector_cpu_targets() -> list[Target]:
        """Returns a list of available vector CPU targets, ordered from least to most capable."""
        return _available_vector_targets()


@cache
def _available_vector_targets() -> list[Target]:
    """Returns available vector targets, sorted from leat to most capable."""

    targets: list[Target] = []

    import platform

    if platform.machine() in ["x86_64", "x86", "AMD64", "i386"]:
        try:
            from cpuinfo import get_cpu_info
        except ImportError:
            warn(
                "Unable to determine available x86 vector CPU targets for this system: "
                "py-cpuinfo is not available.",
                UserWarning,
            )
            return []

        flags = set(get_cpu_info()["flags"])

        if {"sse", "sse2", "ssse3", "sse4_1", "sse4_2"} < flags:
            targets.append(Target.X86_SSE)

        if {"avx", "avx2"} < flags:
            targets.append(Target.X86_AVX)

        if {"avx512f"} < flags:
            targets.append(Target.X86_AVX512)

        if {"avx512_fp16"} < flags:
            targets.append(Target.X86_AVX512_FP16)
    else:
        warn(
            "Unable to determine available vector CPU targets for this system: "
            f"unknown platform {platform.machine()}.",
            UserWarning,
        )

    return targets
