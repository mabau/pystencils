from __future__ import annotations
from typing import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...codegen.target import Target


@dataclass
class CompilerInfo(ABC):
    """Base class for compiler infos."""

    openmp: bool = True
    """Enable/disable OpenMP compilation"""

    optlevel: str | None = "fast"
    """Compiler optimization level"""

    cxx_standard: str = "c++11"
    """C++ language standard to be compiled with"""

    target: Target = Target.CurrentCPU
    """Hardware target to compile for.
    
    Here, `Target.CurrentCPU` represents the current hardware,
    which is reflected by ``-march=native`` in GNU-like compilers.
    """

    @abstractmethod
    def cxx(self) -> str:
        """Path to the executable of this compiler"""

    @abstractmethod
    def cxxflags(self) -> list[str]:
        """Compiler flags affecting C++ compilation"""

    @abstractmethod
    def linker_flags(self) -> list[str]:
        """Flags affecting linkage of the extension module"""

    @abstractmethod
    def include_flags(self, include_dirs: Sequence[str]) -> list[str]:
        """Convert a list of include directories into corresponding compiler flags"""

    @abstractmethod
    def restrict_qualifier(self) -> str:
        """*restrict* memory qualifier recognized by this compiler"""


class _GnuLikeCliCompiler(CompilerInfo):
    def cxxflags(self) -> list[str]:
        flags = ["-DNDEBUG", f"-std={self.cxx_standard}", "-fPIC"]

        if self.optlevel is not None:
            flags.append(f"-O{self.optlevel}")

        if self.openmp:
            flags.append("-fopenmp")

        match self.target:
            case Target.CurrentCPU:
                flags.append("-march=native")
            case Target.X86_SSE:
                flags += ["-march=x86-64-v2"]
            case Target.X86_AVX:
                flags += ["-march=x86-64-v3"]
            case Target.X86_AVX512:
                flags += ["-march=x86-64-v4"]
            case Target.X86_AVX512_FP16:
                flags += ["-march=x86-64-v4", "-mavx512fp16"]

        return flags
    
    def linker_flags(self) -> list[str]:
        return ["-shared"]

    def include_flags(self, include_dirs: Sequence[str]) -> list[str]:
        return [f"-I{d}" for d in include_dirs]

    def restrict_qualifier(self) -> str:
        return "__restrict__"


class GccInfo(_GnuLikeCliCompiler):
    """Compiler info for the GNU Compiler Collection C++ compiler (``g++``)."""

    def cxx(self) -> str:
        return "g++"


@dataclass
class ClangInfo(_GnuLikeCliCompiler):
    """Compiler info for the LLVM C++ compiler (``clang``)."""
    
    llvm_version: int | None = None
    """Major version number of the LLVM installation providing the compiler."""

    def cxx(self) -> str:
        if self.llvm_version is None:
            return "clang"
        else:
            return f"clang-{self.llvm_version}"
        
    def linker_flags(self) -> list[str]:
        return super().linker_flags() + ["-lstdc++"]
