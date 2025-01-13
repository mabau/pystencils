from __future__ import annotations
from typing import cast, Sequence, Iterable, TYPE_CHECKING
from dataclasses import dataclass, replace

from .target import Target
from .config import CreateKernelConfig, OpenMpConfig, VectorizationConfig, AUTO
from .kernel import Kernel, GpuKernel, GpuThreadsRange
from .properties import PsSymbolProperty, FieldShape, FieldStride, FieldBasePtr
from .parameters import Parameter

from ..types import create_numeric_type, PsIntegerType, PsScalarType

from ..backend.memory import PsSymbol
from ..backend.ast import PsAstNode
from ..backend.ast.structural import PsBlock, PsLoop
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
    GenericGpu,
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

        idx_dtype = create_numeric_type(self._cfg.index_dtype)
        assert isinstance(idx_dtype, PsIntegerType)

        self._ctx = KernelCreationContext(
            default_dtype=create_numeric_type(self._cfg.default_dtype),
            index_dtype=idx_dtype,
        )

        self._target = self._cfg.get_target()
        self._platform = self._get_platform()

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

        match self._platform:
            case GenericCpu():
                kernel_ast = self._platform.materialize_iteration_space(
                    kernel_body, self._ctx.get_iteration_space()
                )
            case GenericGpu():
                kernel_ast, gpu_threads = self._platform.materialize_iteration_space(
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
        if self._cfg.target.is_cpu():
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

        if self._cfg.target.is_cpu():
            return create_cpu_kernel_function(
                self._ctx,
                self._platform,
                kernel_ast,
                self._cfg.function_name,
                self._cfg.target,
                self._cfg.get_jit(),
            )
        else:
            return create_gpu_kernel_function(
                self._ctx,
                self._platform,
                kernel_ast,
                gpu_threads,
                self._cfg.function_name,
                self._cfg.target,
                self._cfg.get_jit(),
            )

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

        if self._cfg.index_field is not None:
            ispace = create_sparse_iteration_space(
                self._ctx, assignments, index_field=self._cfg.index_field
            )
        else:
            gls = self._cfg.ghost_layers
            islice = self._cfg.iteration_slice

            if gls is None and islice is None:
                gls = AUTO

            ispace = create_full_iteration_space(
                self._ctx,
                assignments,
                ghost_layers=gls,
                iteration_slice=islice,
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

        cpu_cfg = self._cfg.cpu_optim

        if cpu_cfg is None:
            return kernel_ast

        if cpu_cfg.loop_blocking:
            raise NotImplementedError("Loop blocking not implemented yet.")

        kernel_ast = self._vectorize(kernel_ast)

        if cpu_cfg.openmp is not False:
            from ..backend.transformations import AddOpenMP

            params = (
                cpu_cfg.openmp
                if isinstance(cpu_cfg.openmp, OpenMpConfig)
                else OpenMpConfig()
            )
            add_omp = AddOpenMP(self._ctx, params)
            kernel_ast = cast(PsBlock, add_omp(kernel_ast))

            if self._intermediates is not None:
                self._intermediates.cpu_openmp = kernel_ast.clone()

        if cpu_cfg.use_cacheline_zeroing:
            raise NotImplementedError("CL-zeroing not implemented yet")

        return kernel_ast

    def _vectorize(self, kernel_ast: PsBlock) -> PsBlock:
        assert self._cfg.cpu_optim is not None
        vec_config = self._cfg.cpu_optim.get_vectorization_config()
        if vec_config is None:
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
        if vec_config.assume_inner_stride_one:
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
        if vec_config.lanes is None:
            lanes = VectorizationConfig.default_lanes(
                self._target, cast(PsScalarType, self._ctx.default_dtype)
            )
        else:
            lanes = vec_config.lanes

        vectorizer = LoopVectorizer(self._ctx, lanes)

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

        elif Target._GPU in self._target:
            match self._target:
                case Target.SYCL:
                    from ..backend.platforms import SyclPlatform

                    return SyclPlatform(self._ctx, self._cfg.gpu_indexing)
                case Target.CUDA:
                    from ..backend.platforms import CudaPlatform

                    return CudaPlatform(self._ctx, self._cfg.gpu_indexing)

        raise NotImplementedError(
            f"Code generation for target {self._target} not implemented"
        )


def create_cpu_kernel_function(
    ctx: KernelCreationContext,
    platform: Platform,
    body: PsBlock,
    function_name: str,
    target_spec: Target,
    jit: JitBase,
) -> Kernel:
    undef_symbols = collect_undefined_symbols(body)

    params = _get_function_params(ctx, undef_symbols)
    req_headers = _get_headers(ctx, platform, body)

    kfunc = Kernel(body, target_spec, function_name, params, req_headers, jit)
    kfunc.metadata.update(ctx.metadata)
    return kfunc


def create_gpu_kernel_function(
    ctx: KernelCreationContext,
    platform: Platform,
    body: PsBlock,
    threads_range: GpuThreadsRange | None,
    function_name: str,
    target_spec: Target,
    jit: JitBase,
) -> GpuKernel:
    undef_symbols = collect_undefined_symbols(body)

    if threads_range is not None:
        for threads in threads_range.num_work_items:
            undef_symbols |= collect_undefined_symbols(threads)

    params = _get_function_params(ctx, undef_symbols)
    req_headers = _get_headers(ctx, platform, body)

    kfunc = GpuKernel(
        body,
        threads_range,
        target_spec,
        function_name,
        params,
        req_headers,
        jit,
    )
    kfunc.metadata.update(ctx.metadata)
    return kfunc


def _get_function_params(
    ctx: KernelCreationContext, symbols: Iterable[PsSymbol]
) -> list[Parameter]:
    params: list[Parameter] = []

    from pystencils.backend.memory import BufferBasePtr

    for symb in symbols:
        props: set[PsSymbolProperty] = set()
        for prop in symb.properties:
            match prop:
                case FieldShape() | FieldStride():
                    props.add(prop)
                case BufferBasePtr(buf):
                    field = ctx.find_field(buf.name)
                    props.add(FieldBasePtr(field))
        params.append(Parameter(symb.name, symb.get_dtype(), props))

    params.sort(key=lambda p: p.name)
    return params


def _get_headers(
    ctx: KernelCreationContext, platform: Platform, body: PsBlock
) -> set[str]:
    req_headers = collect_required_headers(body)
    req_headers |= platform.required_headers
    req_headers |= ctx.required_headers
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
