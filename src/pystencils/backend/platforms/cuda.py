from warnings import warn

from ...types import constify
from ..exceptions import MaterializationError
from .generic_gpu import GenericGpu, GpuThreadsRange

from ..kernelcreation import (
    Typifier,
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace,
    AstFactory,
)

from ..kernelcreation.context import KernelCreationContext
from ..ast.structural import PsBlock, PsConditional, PsDeclaration
from ..ast.expressions import (
    PsExpression,
    PsLiteralExpr,
    PsCast,
    PsCall,
    PsLookup,
    PsBufferAcc,
)
from ..ast.expressions import PsLt, PsAnd
from ...types import PsSignedIntegerType, PsIeeeFloatType
from ..literals import PsLiteral
from ..functions import PsMathFunction, MathFunctions, CFunction
from ...config import GpuIndexingConfig

int32 = PsSignedIntegerType(width=32, const=False)

BLOCK_IDX = [
    PsLiteralExpr(PsLiteral(f"blockIdx.{coord}", int32)) for coord in ("x", "y", "z")
]
THREAD_IDX = [
    PsLiteralExpr(PsLiteral(f"threadIdx.{coord}", int32)) for coord in ("x", "y", "z")
]
BLOCK_DIM = [
    PsLiteralExpr(PsLiteral(f"blockDim.{coord}", int32)) for coord in ("x", "y", "z")
]
GRID_DIM = [
    PsLiteralExpr(PsLiteral(f"gridDim.{coord}", int32)) for coord in ("x", "y", "z")
]


class CudaPlatform(GenericGpu):
    """Platform for CUDA-based GPUs."""

    def __init__(
        self, ctx: KernelCreationContext, indexing_cfg: GpuIndexingConfig | None = None
    ) -> None:
        super().__init__(ctx)
        self._cfg = indexing_cfg if indexing_cfg is not None else GpuIndexingConfig()
        self._typify = Typifier(ctx)

    @property
    def required_headers(self) -> set[str]:
        return {'"gpu_defines.h"'}

    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> tuple[PsBlock, GpuThreadsRange | None]:
        if isinstance(ispace, FullIterationSpace):
            return self._prepend_dense_translation(body, ispace)
        elif isinstance(ispace, SparseIterationSpace):
            return self._prepend_sparse_translation(body, ispace)
        else:
            raise MaterializationError(f"Unknown type of iteration space: {ispace}")

    def select_function(self, call: PsCall) -> PsExpression:
        assert isinstance(call.function, PsMathFunction)

        func = call.function.func
        dtype = call.get_dtype()
        arg_types = (dtype,) * func.num_args

        if isinstance(dtype, PsIeeeFloatType):
            match func:
                case (
                    MathFunctions.Exp
                    | MathFunctions.Log
                    | MathFunctions.Sin
                    | MathFunctions.Cos
                    | MathFunctions.Ceil
                    | MathFunctions.Floor
                ) if dtype.width in (16, 32, 64):
                    prefix = "h" if dtype.width == 16 else ""
                    suffix = "f" if dtype.width == 32 else ""
                    name = f"{prefix}{func.function_name}{suffix}"
                    cfunc = CFunction(name, arg_types, dtype)

                case (
                    MathFunctions.Pow
                    | MathFunctions.Tan
                    | MathFunctions.Sinh
                    | MathFunctions.Cosh
                    | MathFunctions.ASin
                    | MathFunctions.ACos
                    | MathFunctions.ATan
                    | MathFunctions.ATan2
                ) if dtype.width in (32, 64):
                    #   These are unavailable for fp16
                    suffix = "f" if dtype.width == 32 else ""
                    name = f"{func.function_name}{suffix}"
                    cfunc = CFunction(name, arg_types, dtype)

                case (
                    MathFunctions.Min | MathFunctions.Max | MathFunctions.Abs
                ) if dtype.width in (32, 64):
                    suffix = "f" if dtype.width == 32 else ""
                    name = f"f{func.function_name}{suffix}"
                    cfunc = CFunction(name, arg_types, dtype)

                case MathFunctions.Abs if dtype.width == 16:
                    cfunc = CFunction(" __habs", arg_types, dtype)

                case _:
                    raise MaterializationError(
                        f"Cannot materialize call to function {func}"
                    )

            call.function = cfunc
            return call

        raise MaterializationError(
            f"No implementation available for function {func} on data type {dtype}"
        )

    #   Internals

    def _prepend_dense_translation(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> tuple[PsBlock, GpuThreadsRange | None]:
        dimensions = ispace.dimensions_in_loop_order()

        if not self._cfg.manual_launch_grid:
            try:
                threads_range = GpuThreadsRange.from_ispace(ispace)
            except MaterializationError as e:
                warn(
                    str(e.args[0])
                    + "\nIf this is intended, set `manual_launch_grid=True` in the code generator configuration.",
                    UserWarning,
                )
                threads_range = None
        else:
            threads_range = None

        indexing_decls = []
        conds = []
        for i, dim in enumerate(dimensions[::-1]):
            dim.counter.dtype = constify(dim.counter.get_dtype())

            ctr = PsExpression.make(dim.counter)
            indexing_decls.append(
                self._typify(
                    PsDeclaration(
                        ctr,
                        dim.start
                        + dim.step
                        * PsCast(ctr.get_dtype(), self._linear_thread_idx(i)),
                    )
                )
            )
            if not self._cfg.omit_range_check:
                conds.append(PsLt(ctr, dim.stop))

        indexing_decls = indexing_decls[::-1]

        if conds:
            condition: PsExpression = conds[0]
            for cond in conds[1:]:
                condition = PsAnd(condition, cond)
            ast = PsBlock(indexing_decls + [PsConditional(condition, body)])
        else:
            body.statements = indexing_decls + body.statements
            ast = body

        return ast, threads_range

    def _prepend_sparse_translation(
        self, body: PsBlock, ispace: SparseIterationSpace
    ) -> tuple[PsBlock, GpuThreadsRange]:
        factory = AstFactory(self._ctx)
        ispace.sparse_counter.dtype = constify(ispace.sparse_counter.get_dtype())

        sparse_ctr = PsExpression.make(ispace.sparse_counter)
        thread_idx = self._linear_thread_idx(0)
        sparse_idx_decl = self._typify(
            PsDeclaration(sparse_ctr, PsCast(sparse_ctr.get_dtype(), thread_idx))
        )

        mappings = [
            PsDeclaration(
                PsExpression.make(ctr),
                PsLookup(
                    PsBufferAcc(
                        ispace.index_list.base_pointer,
                        (sparse_ctr, factory.parse_index(0)),
                    ),
                    coord.name,
                ),
            )
            for ctr, coord in zip(ispace.spatial_indices, ispace.coordinate_members)
        ]
        body.statements = mappings + body.statements

        if not self._cfg.omit_range_check:
            stop = PsExpression.make(ispace.index_list.shape[0])
            condition = PsLt(sparse_ctr, stop)
            ast = PsBlock([sparse_idx_decl, PsConditional(condition, body)])
        else:
            body.statements = [sparse_idx_decl] + body.statements
            ast = body

        return ast, GpuThreadsRange.from_ispace(ispace)

    def _linear_thread_idx(self, coord: int):
        block_size = BLOCK_DIM[coord]
        block_idx = BLOCK_IDX[coord]
        thread_idx = THREAD_IDX[coord]
        return block_idx * block_size + thread_idx
