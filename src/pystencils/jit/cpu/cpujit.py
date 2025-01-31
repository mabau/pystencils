from __future__ import annotations

from types import ModuleType
from pathlib import Path
import subprocess
from copy import copy
from abc import ABC, abstractmethod

from ...codegen.config import _AUTO_TYPE, AUTO

from ..jit import JitError, JitBase, KernelWrapper
from ...codegen import Kernel
from .compiler_info import CompilerInfo, GccInfo


class CpuJit(JitBase):
    """Just-in-time compiler for CPU kernels.

    **Creation**
    
    To configure and create a CPU JIT compiler instance, use the `create` factory method.

    **Implementation Details**

    The `CpuJit` class acts as an orchestrator between two components:
    
    - The *extension module builder* produces the code of the dynamically built extension module
      that contains the kernel and its invocation wrappers;
    - The *compiler info* describes the host compiler used to compile and link that extension module.

    Args:
        compiler_info: The compiler info object defining the capabilities
            and command-line interface of the host compiler
        ext_module_builder: Extension module builder object used to generate the kernel extension module
        objcache: Directory to cache the generated code files and compiled modules in.
            If `None`, a temporary directory will be used, and compilation results will not be cached.
    """

    @staticmethod
    def create(
        compiler_info: CompilerInfo | None = None,
        objcache: str | Path | _AUTO_TYPE | None = AUTO,
    ) -> CpuJit:
        """Configure and create a CPU JIT compiler object.
        
        Args:
            compiler_info: Compiler info object defining capabilities and interface of the host compiler.
                If `None`, a default compiler configuration will be determined from the current OS and runtime
                environment.
            objcache: Directory used for caching compilation results.
                If set to `AUTO`, a persistent cache directory in the current user's home will be used.
                If set to `None`, compilation results will not be cached--this may impact performance.

        Returns:
            The CPU just-in-time compiler.
        """
        
        if objcache is AUTO:
            from appdirs import AppDirs

            dirs = AppDirs(appname="pystencils")
            objcache = Path(dirs.user_cache_dir) / "cpujit"
        elif objcache is not None:
            assert not isinstance(objcache, _AUTO_TYPE)
            objcache = Path(objcache)

        if compiler_info is None:
            compiler_info = GccInfo()

        from .cpujit_pybind11 import Pybind11KernelModuleBuilder

        modbuilder = Pybind11KernelModuleBuilder(compiler_info)

        return CpuJit(compiler_info, modbuilder, objcache)

    def __init__(
        self,
        compiler_info: CompilerInfo,
        ext_module_builder: ExtensionModuleBuilderBase,
        objcache: Path | None,
    ):
        self._compiler_info = copy(compiler_info)
        self._ext_module_builder = ext_module_builder

        self._objcache = objcache

        #   Include Directories

        import sysconfig
        from ...include import get_pystencils_include_path

        include_dirs = [
            sysconfig.get_path("include"),
            get_pystencils_include_path(),
        ] + self._ext_module_builder.include_dirs()

        #   Compiler Flags

        self._cxx = self._compiler_info.cxx()
        self._cxx_fixed_flags = (
            self._compiler_info.cxxflags()
            + self._compiler_info.include_flags(include_dirs)
            + self._compiler_info.linker_flags()
        )

    def compile(self, kernel: Kernel) -> KernelWrapper:
        """Compile the given kernel to an executable function.
        
        Args:
            kernel: The kernel object to be compiled.
        
        Returns:
            Wrapper object around the compiled function
        """

        #   Get the Code
        module_name = f"{kernel.name}_jit"
        cpp_code = self._ext_module_builder.render_module(kernel, module_name)

        #   Get compiler information
        import sysconfig

        so_abi = sysconfig.get_config_var("SOABI")
        lib_suffix = f"{so_abi}.so"

        #   Compute Code Hash
        code_utf8: bytes = cpp_code.encode("utf-8")
        compiler_utf8: bytes = (" ".join([self._cxx] + self._cxx_fixed_flags)).encode("utf-8")
        import hashlib

        module_hash = hashlib.sha256(code_utf8 + compiler_utf8)
        module_stem = f"module_{module_hash.hexdigest()}"

        def compile_and_load(module_dir: Path):
            cpp_file = module_dir / f"{module_stem}.cpp"
            if not cpp_file.exists():
                cpp_file.write_bytes(code_utf8)

            lib_file = module_dir / f"{module_stem}.{lib_suffix}"
            if not lib_file.exists():
                self._compile_extension_module(cpp_file, lib_file)

            module = self._load_extension_module(module_name, lib_file)
            return module

        if self._objcache is not None:
            module_dir = self._objcache
            #   Lock module
            import fasteners

            lockfile = module_dir / f"{module_stem}.lock"
            with fasteners.InterProcessLock(lockfile):
                module = compile_and_load(module_dir)
        else:
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmpdir:
                module_dir = Path(tmpdir)
                module = compile_and_load(module_dir)

        return self._ext_module_builder.get_wrapper(kernel, module)

    def _compile_extension_module(self, src_file: Path, libfile: Path):
        args = (
            [self._cxx]
            + self._cxx_fixed_flags
            + ["-o", str(libfile), str(src_file)]
        )

        result = subprocess.run(args, capture_output=True)
        if result.returncode != 0:
            raise JitError(
                "Compilation failed: C++ compiler terminated with an error.\n"
                + result.stderr.decode()
            )

    def _load_extension_module(self, module_name: str, module_loc: Path) -> ModuleType:
        from importlib import util as iutil

        spec = iutil.spec_from_file_location(name=module_name, location=module_loc)
        if spec is None:
            raise JitError(
                "Unable to load kernel extension module -- this is probably a bug."
            )
        mod = iutil.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod


class ExtensionModuleBuilderBase(ABC):
    """Base class for CPU extension module builders."""

    @staticmethod
    @abstractmethod
    def include_dirs() -> list[str]:
        """List of directories that must be on the include path when compiling
        generated extension modules."""

    @abstractmethod
    def render_module(self, kernel: Kernel, module_name: str) -> str:
        """Produce the extension module code for the given kernel."""

    @abstractmethod
    def get_wrapper(
        self, kernel: Kernel, extension_module: ModuleType
    ) -> KernelWrapper:
        """Produce the invocation wrapper for the given kernel
        and its compiled extension module."""
