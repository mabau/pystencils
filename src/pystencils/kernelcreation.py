from typing import cast, Sequence
from dataclasses import replace

from .target import Target
from .config import (
    CreateKernelConfig,
    OpenMpConfig,
    VectorizationConfig,
)
from .backend import KernelFunction
from .types import create_numeric_type, PsIntegerType, PsScalarType
from .backend.ast.structural import PsBlock, PsLoop
from .backend.kernelcreation import (
    KernelCreationContext,
    KernelAnalysis,
    FreezeExpressions,
    Typifier,
)
from .backend.constants import PsConstant
from .backend.kernelcreation.iteration_space import (
    create_sparse_iteration_space,
    create_full_iteration_space,
    FullIterationSpace,
)
from .backend.platforms import Platform, GenericCpu, GenericVectorCpu, GenericGpu
from .backend.exceptions import VectorizationError

from .backend.transformations import (
    EliminateConstants,
    LowerToC,
    SelectFunctions,
    CanonicalizeSymbols,
    HoistLoopInvariantDeclarations,
)
from .backend.kernelfunction import (
    create_cpu_kernel_function,
    create_gpu_kernel_function,
)

from .simp import AssignmentCollection
from sympy.codegen.ast import AssignmentBase


__all__ = ["create_kernel"]


def create_kernel(
    assignments: AssignmentCollection | Sequence[AssignmentBase] | AssignmentBase,
    config: CreateKernelConfig | None = None,
    **kwargs,
) -> KernelFunction:
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


class DefaultKernelCreationDriver:
    def __init__(self, cfg: CreateKernelConfig):
        self._cfg = cfg

        idx_dtype = create_numeric_type(self._cfg.index_dtype)
        assert isinstance(idx_dtype, PsIntegerType)

        self._ctx = KernelCreationContext(
            default_dtype=create_numeric_type(self._cfg.default_dtype),
            index_dtype=idx_dtype,
        )

        self._target = self._cfg.get_target()
        self._platform = self._get_platform()

    def __call__(
        self,
        assignments: AssignmentCollection | Sequence[AssignmentBase] | AssignmentBase,
    ):
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

        if len(self._ctx.fields.index_fields) > 0 or self._cfg.index_field is not None:
            ispace = create_sparse_iteration_space(
                self._ctx, assignments, index_field=self._cfg.index_field
            )
        else:
            ispace = create_full_iteration_space(
                self._ctx,
                assignments,
                ghost_layers=self._cfg.ghost_layers,
                iteration_slice=self._cfg.iteration_slice,
            )

        self._ctx.set_iteration_space(ispace)

        freeze = FreezeExpressions(self._ctx)
        kernel_body = freeze(assignments)

        typify = Typifier(self._ctx)
        kernel_body = typify(kernel_body)

        match self._platform:
            case GenericCpu():
                kernel_ast = self._platform.materialize_iteration_space(
                    kernel_body, ispace
                )
            case GenericGpu():
                kernel_ast, gpu_threads = self._platform.materialize_iteration_space(
                    kernel_body, ispace
                )

        #   Fold and extract constants
        elim_constants = EliminateConstants(self._ctx, extract_constant_exprs=True)
        kernel_ast = cast(PsBlock, elim_constants(kernel_ast))

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

    def _transform_for_cpu(self, kernel_ast: PsBlock):
        canonicalize = CanonicalizeSymbols(self._ctx, True)
        kernel_ast = cast(PsBlock, canonicalize(kernel_ast))

        hoist_invariants = HoistLoopInvariantDeclarations(self._ctx)
        kernel_ast = cast(PsBlock, hoist_invariants(kernel_ast))

        cpu_cfg = self._cfg.cpu_optim

        if cpu_cfg is None:
            return kernel_ast

        if cpu_cfg.loop_blocking:
            raise NotImplementedError("Loop blocking not implemented yet.")

        kernel_ast = self._vectorize(kernel_ast)

        if cpu_cfg.openmp is not False:
            from .backend.transformations import AddOpenMP

            params = (
                cpu_cfg.openmp
                if isinstance(cpu_cfg.openmp, OpenMpConfig)
                else OpenMpConfig()
            )
            add_omp = AddOpenMP(self._ctx, params)
            kernel_ast = cast(PsBlock, add_omp(kernel_ast))

        if cpu_cfg.use_cacheline_zeroing:
            raise NotImplementedError("CL-zeroing not implemented yet")

        return kernel_ast

    def _vectorize(self, kernel_ast: PsBlock) -> PsBlock:
        assert self._cfg.cpu_optim is not None
        vec_config = self._cfg.cpu_optim.get_vectorization_config()
        if vec_config is None:
            return kernel_ast

        from .backend.transformations import LoopVectorizer, SelectIntrinsics

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

        select_intrin = SelectIntrinsics(self._ctx, self._platform)
        kernel_ast = cast(PsBlock, select_intrin(kernel_ast))

        return kernel_ast

    def _get_platform(self) -> Platform:
        if Target._CPU in self._target:
            if Target._X86 in self._target:
                from .backend.platforms.x86 import X86VectorArch, X86VectorCpu

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
                    from .backend.platforms import SyclPlatform

                    return SyclPlatform(self._ctx, self._cfg.gpu_indexing)
                case Target.CUDA:
                    from .backend.platforms import CudaPlatform

                    return CudaPlatform(self._ctx, self._cfg.gpu_indexing)

        raise NotImplementedError(
            f"Code generation for target {self._target} not implemented"
        )


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
