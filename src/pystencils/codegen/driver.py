from __future__ import annotations
from typing import cast, Sequence, Callable, TYPE_CHECKING
from dataclasses import dataclass, replace

from .target import Target
from .config import (
    CreateKernelConfig,
    VectorizationOptions,
    AUTO,
    _AUTO_TYPE,
    GhostLayerSpec,
    IterationSliceSpec,
    GpuIndexingScheme,
    GpuOptions,
)
from .kernel import Kernel, GpuKernel
from .properties import PsSymbolProperty, FieldBasePtr
from .parameters import Parameter
from .functions import Lambda
from .gpu_indexing import GpuIndexing, GpuLaunchConfiguration

from ..field import Field
from ..types import PsIntegerType, PsScalarType

from ..backend.memory import PsSymbol
from ..backend.ast import PsAstNode
from ..backend.ast.structural import PsBlock, PsLoop
from ..backend.ast.expressions import PsExpression
from ..backend.ast.analysis import collect_undefined_symbols, collect_required_headers
from ..backend.kernelcreation import (
    KernelCreationContext,
    KernelAnalysis,
    FreezeExpressions,
    Typifier,
)
from ..backend.constants import PsConstant
from ..backend.kernelcreation.iteration_space import (
    create_sparse_iteration_space,
    create_full_iteration_space,
    FullIterationSpace,
)
from ..backend.platforms import (
    Platform,
    GenericCpu,
    GenericVectorCpu,
)
from ..backend.exceptions import VectorizationError

from ..backend.transformations import (
    EliminateConstants,
    LowerToC,
    SelectFunctions,
    CanonicalizeSymbols,
    HoistLoopInvariantDeclarations,
)

from ..simp import AssignmentCollection
from sympy.codegen.ast import AssignmentBase

if TYPE_CHECKING:
    from ..jit import JitBase


__all__ = ["create_kernel"]


def create_kernel(
    assignments: AssignmentCollection | Sequence[AssignmentBase] | AssignmentBase,
    config: CreateKernelConfig | None = None,
    **kwargs,
) -> Kernel:
    """Create a kernel function from a set of assignments.

    Args:
        assignments: The kernel's sequence of assignments, expressed using SymPy
        config: The configuration for the kernel translator
        kwargs: If ``config`` is not set, it is created from the keyword arguments;
            if it is set, its option will be overridden by any keyword arguments.

    Returns:
        The numerical kernel in pystencil's internal representation, ready to be
        exported or compiled
    """

    if not config:
        config = CreateKernelConfig()

    if kwargs:
        config = replace(config, **kwargs)

    driver = DefaultKernelCreationDriver(config)
    return driver(assignments)


def get_driver(
    cfg: CreateKernelConfig, *, retain_intermediates: bool = False
) -> DefaultKernelCreationDriver:
    """Create a code generation driver object from the given configuration.

    Args:
        cfg: Configuration for the code generator
        retain_intermediates: If `True`, instructs the driver to keep copies of
            the intermediate results of its stages for later inspection.
    """
    return DefaultKernelCreationDriver(cfg, retain_intermediates)


class DefaultKernelCreationDriver:
    """Drives the default kernel creation sequence.

    Args:
        cfg: Configuration for the code generator
        retain_intermediates: If `True`, instructs the driver to keep copies of
            the intermediate results of its stages for later inspection.
    """

    def __init__(self, cfg: CreateKernelConfig, retain_intermediates: bool = False):
        self._cfg = cfg

        #   Data Type Options
        idx_dtype: PsIntegerType = cfg.get_option("index_dtype")
        default_dtype: PsScalarType = cfg.get_option("default_dtype")

        #   Iteration Space Options
        num_ispace_options_set = (
            int(cfg.is_option_set("ghost_layers"))
            + int(cfg.is_option_set("iteration_slice"))
            + int(cfg.is_option_set("index_field"))
        )

        if num_ispace_options_set > 1:
            raise ValueError(
                "At most one of the options 'ghost_layers' 'iteration_slice' and 'index_field' may be set."
            )

        self._ghost_layers: GhostLayerSpec | None = cfg.get_option("ghost_layers")
        self._iteration_slice: IterationSliceSpec | None = cfg.get_option(
            "iteration_slice"
        )
        self._index_field: Field | None = cfg.get_option("index_field")

        if num_ispace_options_set == 0:
            self._ghost_layers = AUTO

        #   Create the context
        self._ctx = KernelCreationContext(
            default_dtype=default_dtype,
            index_dtype=idx_dtype,
        )

        self._target = cfg.get_target()
        self._gpu_indexing: GpuIndexing | None = self._get_gpu_indexing()
        self._platform = self._get_platform()

        self._intermediates: CodegenIntermediates | None
        if retain_intermediates:
            self._intermediates = CodegenIntermediates()
        else:
            self._intermediates = None

    @property
    def intermediates(self) -> CodegenIntermediates | None:
        return self._intermediates

    def __call__(
        self,
        assignments: AssignmentCollection | Sequence[AssignmentBase] | AssignmentBase,
    ) -> Kernel:
        kernel_body = self.parse_kernel_body(assignments)

        kernel_ast = self._platform.materialize_iteration_space(
            kernel_body, self._ctx.get_iteration_space()
        )

        if self._intermediates is not None:
            self._intermediates.materialized_ispace = kernel_ast.clone()

        #   Fold and extract constants
        elim_constants = EliminateConstants(self._ctx, extract_constant_exprs=True)
        kernel_ast = cast(PsBlock, elim_constants(kernel_ast))

        if self._intermediates is not None:
            self._intermediates.constants_eliminated = kernel_ast.clone()

        #   Target-Specific optimizations
        if self._target.is_cpu():
            kernel_ast = self._transform_for_cpu(kernel_ast)

        #   Note: After this point, the AST may contain intrinsics, so type-dependent
        #   transformations cannot be run any more

        #   Lowering
        lower_to_c = LowerToC(self._ctx)
        kernel_ast = cast(PsBlock, lower_to_c(kernel_ast))

        select_functions = SelectFunctions(self._platform)
        kernel_ast = cast(PsBlock, select_functions(kernel_ast))

        if self._intermediates is not None:
            self._intermediates.lowered = kernel_ast.clone()

        #   Late canonicalization pass: Canonicalize new symbols introduced by LowerToC

        canonicalize = CanonicalizeSymbols(self._ctx, True)
        kernel_ast = cast(PsBlock, canonicalize(kernel_ast))

        kernel_factory = KernelFactory(self._ctx)

        if self._target.is_cpu() or self._target == Target.SYCL:
            return kernel_factory.create_generic_kernel(
                self._platform,
                kernel_ast,
                self._cfg.get_option("function_name"),
                self._target,
                self._cfg.get_jit(),
            )
        elif self._target.is_gpu():
            assert self._gpu_indexing is not None

            return kernel_factory.create_gpu_kernel(
                self._platform,
                kernel_ast,
                self._cfg.get_option("function_name"),
                self._target,
                self._cfg.get_jit(),
                self._gpu_indexing.get_launch_config_factory(),
            )
        else:
            assert False, "unexpected target"

    def parse_kernel_body(
        self,
        assignments: AssignmentCollection | Sequence[AssignmentBase] | AssignmentBase,
    ) -> PsBlock:
        if isinstance(assignments, AssignmentBase):
            assignments = [assignments]

        if not isinstance(assignments, AssignmentCollection):
            assignments = AssignmentCollection(assignments)  # type: ignore

        _ = _parse_simplification_hints(assignments)

        analysis = KernelAnalysis(
            self._ctx,
            not self._cfg.skip_independence_check,
            not self._cfg.allow_double_writes,
        )
        analysis(assignments)

        if self._index_field is not None:
            ispace = create_sparse_iteration_space(
                self._ctx, assignments, index_field=self._cfg.index_field
            )
        else:
            gls: GhostLayerSpec | None
            if self._ghost_layers == AUTO:
                infer_gls = True
                gls = None
            else:
                assert not isinstance(self._ghost_layers, _AUTO_TYPE)
                infer_gls = False
                gls = self._ghost_layers

            ispace = create_full_iteration_space(
                self._ctx,
                assignments,
                ghost_layers=gls,
                iteration_slice=self._iteration_slice,
                infer_ghost_layers=infer_gls,
            )

        self._ctx.set_iteration_space(ispace)

        freeze = FreezeExpressions(self._ctx)
        kernel_body = freeze(assignments)

        typify = Typifier(self._ctx)
        kernel_body = typify(kernel_body)

        if self._intermediates is not None:
            self._intermediates.parsed_body = kernel_body.clone()

        return kernel_body

    def _transform_for_cpu(self, kernel_ast: PsBlock) -> PsBlock:
        canonicalize = CanonicalizeSymbols(self._ctx, True)
        kernel_ast = cast(PsBlock, canonicalize(kernel_ast))

        if self._intermediates is not None:
            self._intermediates.cpu_canonicalize = kernel_ast.clone()

        hoist_invariants = HoistLoopInvariantDeclarations(self._ctx)
        kernel_ast = cast(PsBlock, hoist_invariants(kernel_ast))

        if self._intermediates is not None:
            self._intermediates.cpu_hoist_invariants = kernel_ast.clone()

        cpu_cfg = self._cfg.cpu

        if cpu_cfg is None:
            return kernel_ast

        if cpu_cfg.loop_blocking:
            raise NotImplementedError("Loop blocking not implemented yet.")

        kernel_ast = self._vectorize(kernel_ast)
        kernel_ast = self._add_openmp(kernel_ast)

        if cpu_cfg.use_cacheline_zeroing:
            raise NotImplementedError("CL-zeroing not implemented yet")

        return kernel_ast

    def _add_openmp(self, kernel_ast: PsBlock) -> PsBlock:
        omp_options = self._cfg.cpu.openmp
        enable_omp: bool = omp_options.get_option("enable")

        if enable_omp:
            from ..backend.transformations import AddOpenMP

            add_omp = AddOpenMP(
                self._ctx,
                nesting_depth=omp_options.get_option("nesting_depth"),
                num_threads=omp_options.get_option("num_threads"),
                schedule=omp_options.get_option("schedule"),
                collapse=omp_options.get_option("collapse"),
                omit_parallel=omp_options.get_option("omit_parallel_construct"),
            )
            kernel_ast = cast(PsBlock, add_omp(kernel_ast))

            if self._intermediates is not None:
                self._intermediates.cpu_openmp = kernel_ast.clone()

        return kernel_ast

    def _vectorize(self, kernel_ast: PsBlock) -> PsBlock:
        vec_options = self._cfg.cpu.vectorize

        enable_vec = vec_options.get_option("enable")

        if not enable_vec:
            return kernel_ast

        from ..backend.transformations import LoopVectorizer, SelectIntrinsics

        assert isinstance(self._platform, GenericVectorCpu)

        ispace = self._ctx.get_iteration_space()
        if not isinstance(ispace, FullIterationSpace):
            raise VectorizationError(
                "Unable to vectorize kernel: The kernel is not using a dense iteration space."
            )

        inner_loop_coord = ispace.loop_order[-1]
        inner_loop_dim = ispace.dimensions[inner_loop_coord]

        #   Apply stride (TODO: and alignment) assumptions
        assume_unit_stride: bool = vec_options.get_option("assume_inner_stride_one")

        if assume_unit_stride:
            for field in self._ctx.fields:
                buf = self._ctx.get_buffer(field)
                inner_stride = buf.strides[inner_loop_coord]
                if isinstance(inner_stride, PsConstant):
                    if inner_stride.value != 1:
                        raise VectorizationError(
                            f"Unable to apply assumption 'assume_inner_stride_one': "
                            f"Field {field} has fixed stride {inner_stride} "
                            f"set in the inner coordinate {inner_loop_coord}."
                        )
                else:
                    buf.strides[inner_loop_coord] = PsConstant(1, buf.index_type)
                    #   TODO: Communicate assumption to runtime system via a precondition

        #   Call loop vectorizer
        num_lanes: int | None = vec_options.get_option("lanes")

        if num_lanes is None:
            num_lanes = VectorizationOptions.default_lanes(
                self._target, cast(PsScalarType, self._ctx.default_dtype)
            )

        vectorizer = LoopVectorizer(self._ctx, num_lanes)

        def loop_predicate(loop: PsLoop):
            return loop.counter.symbol == inner_loop_dim.counter

        kernel_ast = vectorizer.vectorize_select_loops(kernel_ast, loop_predicate)

        if self._intermediates is not None:
            self._intermediates.cpu_vectorize = kernel_ast.clone()

        select_intrin = SelectIntrinsics(self._ctx, self._platform)
        kernel_ast = cast(PsBlock, select_intrin(kernel_ast))

        if self._intermediates is not None:
            self._intermediates.cpu_select_intrins = kernel_ast.clone()

        return kernel_ast

    def _get_gpu_indexing(self) -> GpuIndexing | None:
        if self._target != Target.CUDA:
            return None

        idx_scheme: GpuIndexingScheme = self._cfg.gpu.get_option("indexing_scheme")
        manual_launch_grid: bool = self._cfg.gpu.get_option("manual_launch_grid")
        assume_warp_aligned_block_size: bool = self._cfg.gpu.get_option("assume_warp_aligned_block_size")
        warp_size: int | None = self._cfg.gpu.get_option("warp_size")

        if warp_size is None:
            warp_size = GpuOptions.default_warp_size(self._target)

        return GpuIndexing(
            self._ctx,
            self._target,
            idx_scheme,
            warp_size,
            manual_launch_grid,
            assume_warp_aligned_block_size,
        )

    def _get_platform(self) -> Platform:
        if Target._CPU in self._target:
            if Target._X86 in self._target:
                from ..backend.platforms.x86 import X86VectorArch, X86VectorCpu

                arch: X86VectorArch

                if Target._SSE in self._target:
                    arch = X86VectorArch.SSE
                elif Target._AVX in self._target:
                    arch = X86VectorArch.AVX
                elif Target._AVX512 in self._target:
                    if Target._FP16 in self._target:
                        arch = X86VectorArch.AVX512_FP16
                    else:
                        arch = X86VectorArch.AVX512
                else:
                    assert False, "unreachable code"

                return X86VectorCpu(self._ctx, arch)
            elif self._target == Target.GenericCPU:
                return GenericCpu(self._ctx)
            else:
                raise NotImplementedError(
                    f"No platform is currently available for CPU target {self._target}"
                )

        elif self._target.is_gpu():
            match self._target:
                case Target.CUDA:
                    from ..backend.platforms import CudaPlatform

                    thread_mapping = (
                        self._gpu_indexing.get_thread_mapping()
                        if self._gpu_indexing is not None
                        else None
                    )

                    return CudaPlatform(
                        self._ctx,
                        thread_mapping=thread_mapping,
                    )
        elif self._target == Target.SYCL:
            from ..backend.platforms import SyclPlatform

            auto_block_size: bool = self._cfg.sycl.get_option("automatic_block_size")

            return SyclPlatform(
                self._ctx,
                automatic_block_size=auto_block_size,
            )

        raise NotImplementedError(
            f"Code generation for target {self._target} not implemented"
        )


class KernelFactory:
    """Factory for wrapping up backend and IR objects into exportable kernels and function objects."""

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    def create_lambda(self, expr: PsExpression) -> Lambda:
        """Create a Lambda from an expression."""
        params = self._get_function_params(expr)
        return Lambda(expr, params)

    def create_generic_kernel(
        self,
        platform: Platform,
        body: PsBlock,
        function_name: str,
        target_spec: Target,
        jit: JitBase,
    ) -> Kernel:
        """Create a kernel for a generic target"""
        params = self._get_function_params(body)
        req_headers = self._get_headers(platform, body)

        kfunc = Kernel(body, target_spec, function_name, params, req_headers, jit)
        kfunc.metadata.update(self._ctx.metadata)
        return kfunc

    def create_gpu_kernel(
        self,
        platform: Platform,
        body: PsBlock,
        function_name: str,
        target_spec: Target,
        jit: JitBase,
        launch_config_factory: Callable[[], GpuLaunchConfiguration],
    ) -> GpuKernel:
        """Create a kernel for a GPU target"""
        params = self._get_function_params(body)
        req_headers = self._get_headers(platform, body)

        kfunc = GpuKernel(
            body,
            target_spec,
            function_name,
            params,
            req_headers,
            jit,
            launch_config_factory,
        )
        kfunc.metadata.update(self._ctx.metadata)
        return kfunc

    def _symbol_to_param(self, symbol: PsSymbol):
        from pystencils.backend.memory import BufferBasePtr, BackendPrivateProperty

        props: set[PsSymbolProperty] = set()
        for prop in symbol.properties:
            match prop:
                case BufferBasePtr(buf):
                    field = self._ctx.find_field(buf.name)
                    props.add(FieldBasePtr(field))
                case BackendPrivateProperty():
                    pass
                case _:
                    props.add(prop)

        return Parameter(symbol.name, symbol.get_dtype(), props)

    def _get_function_params(self, ast: PsAstNode) -> list[Parameter]:
        symbols = collect_undefined_symbols(ast)
        params: list[Parameter] = [self._symbol_to_param(s) for s in symbols]
        params.sort(key=lambda p: p.name)
        return params

    def _get_headers(self, platform: Platform, body: PsBlock) -> set[str]:
        req_headers = collect_required_headers(body)
        req_headers |= platform.required_headers
        req_headers |= self._ctx.required_headers
        return req_headers


@dataclass
class StageResult:
    ast: PsAstNode
    label: str


class StageResultSlot:
    def __init__(self, description: str | None = None):
        self._description = description
        self._name: str
        self._lookup: str

    def __set_name__(self, owner, name: str):
        self._name = name
        self._lookup = f"_{name}"

    def __get__(self, obj, objtype=None) -> StageResult | None:
        if obj is None:
            return None

        ast = getattr(obj, self._lookup, None)
        if ast is not None:
            descr = self._name if self._description is None else self._description
            return StageResult(ast, descr)
        else:
            return None

    def __set__(self, obj, val: PsAstNode | None):
        setattr(obj, self._lookup, val)


class CodegenIntermediates:
    """Intermediate results produced by the code generator."""

    parsed_body = StageResultSlot("Freeze & Type Deduction")
    materialized_ispace = StageResultSlot("Iteration Space Materialization")
    constants_eliminated = StageResultSlot("Constant Elimination")
    cpu_canonicalize = StageResultSlot("CPU: Symbol Canonicalization")
    cpu_hoist_invariants = StageResultSlot("CPU: Hoisting of Loop Invariants")
    cpu_vectorize = StageResultSlot("CPU: Vectorization")
    cpu_select_intrins = StageResultSlot("CPU: Intrinsics Selection")
    cpu_openmp = StageResultSlot("CPU: OpenMP Instrumentation")
    lowered = StageResultSlot("C Language Lowering")

    @property
    def available_stages(self) -> Sequence[StageResult]:
        all_results: list[StageResult | None] = [
            getattr(self, name)
            for name, slot in CodegenIntermediates.__dict__.items()
            if isinstance(slot, StageResultSlot)
        ]
        return tuple(filter(lambda r: r is not None, all_results))  # type: ignore


def create_staggered_kernel(
    assignments, target: Target = Target.CPU, gpu_exclusive_conditions=False, **kwargs
):
    raise NotImplementedError(
        "Staggered kernels are not yet implemented for pystencils 2.0"
    )


#   Internals


def _parse_simplification_hints(ac: AssignmentCollection):
    if "split_groups" in ac.simplification_hints:
        raise NotImplementedError(
            "Loop splitting was requested, but is not implemented yet"
        )
