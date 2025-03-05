from __future__ import annotations
from textwrap import indent
from typing import cast, overload

from dataclasses import dataclass

from ...types import PsType, PsVectorType, PsBoolType, PsScalarType

from ..kernelcreation import KernelCreationContext, AstFactory
from ..memory import PsSymbol
from ..constants import PsConstant
from ..functions import PsMathFunction

from ..ast import PsAstNode
from ..ast.structural import (
    PsBlock,
    PsDeclaration,
    PsAssignment,
    PsLoop,
    PsEmptyLeafMixIn,
    PsStructuralNode,
)
from ..ast.expressions import (
    PsExpression,
    PsAddressOf,
    PsCast,
    PsUnOp,
    PsBinOp,
    PsSymbolExpr,
    PsConstantExpr,
    PsLiteral,
    PsCall,
    PsMemAcc,
    PsBufferAcc,
    PsSubscript,
    PsAdd,
    PsMul,
    PsSub,
    PsNeg,
    PsDiv,
)
from ..ast.vector import PsVectorOp, PsVecBroadcast, PsVecMemAcc
from ..ast.analysis import UndefinedSymbolsCollector

from ..exceptions import PsInternalCompilerError, VectorizationError


@dataclass(frozen=True)
class VectorizationAxis:
    """Information about the iteration axis along which a subtree is being vectorized."""

    counter: PsSymbol
    """Scalar iteration counter of this axis"""

    vectorized_counter: PsSymbol | None = None
    """Vectorized iteration counter of this axis"""

    step: PsExpression = PsExpression.make(PsConstant(1))
    """Step size of the scalar iteration"""

    def get_vectorized_counter(self) -> PsSymbol:
        if self.vectorized_counter is None:
            raise PsInternalCompilerError(
                "No vectorized counter defined on this vectorization axis"
            )

        return self.vectorized_counter


class VectorizationContext:
    """Context information for AST vectorization.

    Args:
        lanes: Number of vector lanes
        axis: Iteration axis along which code is being vectorized
    """

    def __init__(
        self,
        ctx: KernelCreationContext,
        lanes: int,
        axis: VectorizationAxis,
        vectorized_symbols: dict[PsSymbol, PsSymbol] | None = None,
    ) -> None:
        self._ctx = ctx
        self._lanes = lanes
        self._axis: VectorizationAxis = axis
        self._vectorized_symbols: dict[PsSymbol, PsSymbol] = (
            {**vectorized_symbols} if vectorized_symbols is not None else dict()
        )
        self._lane_mask: PsSymbol | None = None

        if axis.vectorized_counter is not None:
            self._vectorized_symbols[axis.counter] = axis.vectorized_counter

    @property
    def lanes(self) -> int:
        """Number of vector lanes"""
        return self._lanes

    @property
    def axis(self) -> VectorizationAxis:
        """Iteration axis along which to vectorize"""
        return self._axis

    @property
    def vectorized_symbols(self) -> dict[PsSymbol, PsSymbol]:
        """Dictionary mapping scalar symbols that are being vectorized to their vectorized copies"""
        return self._vectorized_symbols

    @property
    def lane_mask(self) -> PsSymbol | None:
        """Symbol representing the current lane execution mask, or ``None`` if all lanes are active."""
        return self._lane_mask

    @lane_mask.setter
    def lane_mask(self, mask: PsSymbol | None):
        self._lane_mask = mask

    def get_lane_mask_expr(self) -> PsExpression:
        """Retrieve an expression representing the current lane execution mask."""
        if self._lane_mask is not None:
            return PsExpression.make(self._lane_mask)
        else:
            return PsExpression.make(
                PsConstant(True, PsVectorType(PsBoolType(), self._lanes))
            )

    def vectorize_symbol(self, symb: PsSymbol) -> PsSymbol:
        """Vectorize the given symbol of scalar type.

        Creates a duplicate of the given symbol with vectorized data type,
        adds it to the ``vectorized_symbols`` dict,
        and returns the duplicate.

        Raises:
            VectorizationError: If the symbol's data type was not a `PsScalarType`,
                or if the symbol was already vectorized
        """
        if symb in self._vectorized_symbols:
            raise VectorizationError(f"Symbol {symb} was already vectorized.")

        vec_type = self.vector_type(symb.get_dtype())
        vec_symb = self._ctx.duplicate_symbol(symb, vec_type)
        self._vectorized_symbols[symb] = vec_symb
        return vec_symb

    def vector_type(self, scalar_type: PsType) -> PsVectorType:
        """Vectorize the given scalar data type.

        Raises:
            VectorizationError: If the given data type was not a `PsScalarType`.
        """
        if not isinstance(scalar_type, PsScalarType):
            raise VectorizationError(
                f"Unable to vectorize type {scalar_type}: was not a scalar numeric type"
            )
        return PsVectorType(scalar_type, self._lanes)

    def axis_ctr_dependees(self, symbols: set[PsSymbol]) -> set[PsSymbol]:
        """Returns all symbols in `symbols` that depend on the axis counter."""
        return symbols & (self.vectorized_symbols.keys() | {self.axis.counter})


@dataclass
class Affine:
    coeff: PsExpression
    offset: PsExpression

    def __neg__(self):
        return Affine(-self.coeff, -self.offset)

    def __add__(self, other: Affine):
        return Affine(self.coeff + other.coeff, self.offset + other.offset)

    def __sub__(self, other: Affine):
        return Affine(self.coeff - other.coeff, self.offset - other.offset)

    def __mul__(self, factor: PsExpression):
        if not isinstance(factor, PsExpression):
            return NotImplemented
        return Affine(self.coeff * factor, self.offset * factor)

    def __rmul__(self, factor: PsExpression):
        if not isinstance(factor, PsExpression):
            return NotImplemented
        return Affine(self.coeff * factor, self.offset * factor)

    def __truediv__(self, divisor: PsExpression):
        if not isinstance(divisor, PsExpression):
            return NotImplemented
        return Affine(self.coeff / divisor, self.offset / divisor)


class AstVectorizer:
    """Transform a scalar subtree into a SIMD-parallel version of itself.

    The `AstVectorizer` constructs a vectorized copy of a subtree by creating a SIMD-parallel
    version of each of its nodes, one at a time.
    It relies on information given in a `VectorizationContext` that defines the current environment,
    including the vectorization axis, the number of vector lanes, and an execution mask determining
    which vector lanes are active.

    **Memory Accesses:**
    The AST vectorizer is capable of vectorizing `PsMemAcc` and `PsBufferAcc` only under certain circumstances:

    - If all indices are independent of both the vectorization axis' counter and any vectorized symbols,
      the memory access is *lane-invariant*, and its result will be broadcast to all vector lanes.
    - If at most one index depends on the axis counter via an affine expression, and does not depend on any
      vectorized symbols, the memory access can be performed in parallel, either contiguously or strided,
      and is replaced by a `PsVecMemAcc`.
    - All other cases cause vectorization to fail.

    **Legality:**
    The AST vectorizer performs no legality checks and in particular assumes the absence of loop-carried
    dependencies; i.e. all iterations of the vectorized subtree must already be independent of each
    other, and insensitive to execution order.

    **Result and Failures:**
    The AST vectorizer does not alter the original subtree, but constructs and returns a copy of it.
    Any symbols declared within the subtree are therein replaced by canonically renamed,
    vectorized copies of themselves.

    If the AST vectorizer is unable to transform a subtree, it raises a `VectorizationError`.
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx
        self._factory = AstFactory(ctx)
        self._collect_symbols = UndefinedSymbolsCollector()

        from ..kernelcreation import Typifier
        from .eliminate_constants import EliminateConstants
        from .lower_to_c import LowerToC

        self._typifiy = Typifier(ctx)
        self._fold_constants = EliminateConstants(ctx)
        self._lower_to_c = LowerToC(ctx)

    @overload
    def __call__(self, node: PsBlock, vc: VectorizationContext) -> PsBlock:
        pass

    @overload
    def __call__(self, node: PsDeclaration, vc: VectorizationContext) -> PsDeclaration:
        pass

    @overload
    def __call__(self, node: PsAssignment, vc: VectorizationContext) -> PsAssignment:
        pass

    @overload
    def __call__(self, node: PsExpression, vc: VectorizationContext) -> PsExpression:
        pass

    @overload
    def __call__(self, node: PsAstNode, vc: VectorizationContext) -> PsAstNode:
        pass

    def __call__(self, node: PsAstNode, vc: VectorizationContext) -> PsAstNode:
        """Perform subtree vectorization.

        Args:
            node: Root of the subtree that should be vectorized
            vc: Object describing the current vectorization context

        Raises:
            VectorizationError: If a node cannot be vectorized
        """
        return self.visit(node, vc)

    @overload
    def visit(self, node: PsStructuralNode, vc: VectorizationContext) -> PsStructuralNode:
        pass

    @overload
    def visit(self, node: PsExpression, vc: VectorizationContext) -> PsExpression:
        pass

    @overload
    def visit(self, node: PsAstNode, vc: VectorizationContext) -> PsAstNode:
        pass

    def visit(self, node: PsAstNode, vc: VectorizationContext) -> PsAstNode:
        """Vectorize a subtree."""

        match node:
            case PsBlock(stmts):
                return PsBlock([self.visit(n, vc) for n in stmts])

            case PsExpression():
                return self.visit_expr(node, vc)

            case PsDeclaration(_, rhs):
                vec_symb = vc.vectorize_symbol(node.declared_symbol)
                vec_lhs = PsExpression.make(vec_symb)
                vec_rhs = self.visit_expr(rhs, vc)
                return PsDeclaration(vec_lhs, vec_rhs)

            case PsAssignment(lhs, rhs):
                if (
                    isinstance(lhs, PsSymbolExpr)
                    and lhs.symbol in vc.vectorized_symbols
                ):
                    return PsAssignment(
                        self.visit_expr(lhs, vc), self.visit_expr(rhs, vc)
                    )

                if not isinstance(lhs, (PsMemAcc, PsBufferAcc)):
                    raise VectorizationError(f"Unable to vectorize assignment to {lhs}")

                lhs_vec = self.visit_expr(lhs, vc)
                if not isinstance(lhs_vec, PsVecMemAcc):
                    raise VectorizationError(
                        f"Unable to vectorize memory write {node}:\n"
                        f"Index did not depend on axis counter."
                    )

                rhs_vec = self.visit_expr(rhs, vc)
                return PsAssignment(lhs_vec, rhs_vec)

            case PsLoop(counter, start, stop, step, body):
                # Check that loop bounds are lane-invariant
                free_symbols = (
                    self._collect_symbols(start)
                    | self._collect_symbols(stop)
                    | self._collect_symbols(step)
                )
                vec_dependencies = vc.axis_ctr_dependees(free_symbols)
                if vec_dependencies:
                    raise VectorizationError(
                        "Unable to vectorize loop depending on vectorized symbols:\n"
                        f"  Offending dependencies:\n"
                        f"    {vec_dependencies}\n"
                        f"  Found in loop:\n"
                        f"{indent(str(node), '   ')}"
                    )

                vectorized_body = cast(PsBlock, self.visit(body, vc))
                return PsLoop(counter, start, stop, step, vectorized_body)

            case PsEmptyLeafMixIn():
                return node

            case _:
                raise NotImplementedError(f"Vectorization of {node} is not implemented")

    def visit_expr(self, expr: PsExpression, vc: VectorizationContext) -> PsExpression:
        """Vectorize an expression."""

        vec_expr: PsExpression
        scalar_type = expr.get_dtype()

        match expr:
            #   Invalids
            case PsVectorOp() | PsAddressOf():
                raise VectorizationError(f"Unable to vectorize {type(expr)}: {expr}")

            #   Symbols
            case PsSymbolExpr(symb) if symb in vc.vectorized_symbols:
                # Vectorize symbol
                vector_symb = vc.vectorized_symbols[symb]
                vec_expr = PsSymbolExpr(vector_symb)

            case PsSymbolExpr(symb) if symb == vc.axis.counter:
                raise VectorizationError(
                    f"Unable to vectorize occurence of axis counter {symb} "
                    "since no vectorized version of the counter was present in the context."
                )

            #   Symbols, constants, and literals that can be broadcast
            case PsSymbolExpr() | PsConstantExpr() | PsLiteral():
                if isinstance(expr.dtype, PsScalarType):
                    #   Broadcast constant or non-vectorized scalar symbol
                    vec_expr = PsVecBroadcast(vc.lanes, expr.clone())
                else:
                    #   Cannot vectorize non-scalar constants or symbols
                    raise VectorizationError(
                        f"Unable to vectorize expression {expr} of non-scalar data type {expr.dtype}"
                    )

            #   Unary Ops
            case PsCast(target_type, operand):
                vec_expr = PsCast(
                    vc.vector_type(target_type), self.visit_expr(operand, vc)
                )

            case PsUnOp(operand):
                vec_expr = type(expr)(self.visit_expr(operand, vc))

            #   Binary Ops
            case PsBinOp(op1, op2):
                vec_expr = type(expr)(
                    self.visit_expr(op1, vc), self.visit_expr(op2, vc)
                )

            #   Math Functions
            case PsCall(PsMathFunction(func), func_args):
                vec_expr = PsCall(
                    PsMathFunction(func),
                    [self.visit_expr(arg, vc) for arg in func_args],
                )

            #   Other Functions
            case PsCall(func, _):
                raise VectorizationError(
                    f"Unable to vectorize function call to {func}."
                )

            #   Memory Accesses
            case PsMemAcc(ptr, offset):
                if not isinstance(ptr, PsSymbolExpr):
                    raise VectorizationError(
                        f"Unable to vectorize memory access by non-symbol pointer {ptr}"
                    )

                idx_affine = self._index_as_affine(offset, vc)
                if idx_affine is None:
                    vec_expr = PsVecBroadcast(vc.lanes, expr.clone())
                else:
                    stride: PsExpression | None = self._fold_constants(
                        self._typifiy(idx_affine.coeff * vc.axis.step)
                    )

                    if (
                        isinstance(stride, PsConstantExpr)
                        and stride.constant.value == 1
                    ):
                        #   Contiguous access
                        stride = None

                    vec_expr = PsVecMemAcc(
                        ptr.clone(), offset.clone(), vc.lanes, stride
                    )

            case PsBufferAcc(ptr, indices):
                buf = expr.buffer

                ctr_found = False
                access_stride: PsExpression | None = None

                for i, idx in enumerate(indices):
                    idx_affine = self._index_as_affine(idx, vc)
                    if idx_affine is not None:
                        if ctr_found:
                            raise VectorizationError(
                                f"Unable to vectorize buffer access {expr}: "
                                f"Found multiple indices that depend on iteration counter {vc.axis.counter}."
                            )

                        ctr_found = True

                        access_stride = stride = self._fold_constants(
                            self._typifiy(
                                idx_affine.coeff
                                * vc.axis.step
                                * PsExpression.make(buf.strides[i])
                            )
                        )

                if ctr_found:
                    #   Buffer access must be vectorized
                    assert access_stride is not None

                    if (
                        isinstance(access_stride, PsConstantExpr)
                        and access_stride.constant.value == 1
                    ):
                        #   Contiguous access
                        access_stride = None

                    linearized_acc = self._lower_to_c(expr)
                    assert isinstance(linearized_acc, PsMemAcc)

                    vec_expr = PsVecMemAcc(
                        ptr.clone(),
                        linearized_acc.offset.clone(),
                        vc.lanes,
                        access_stride,
                    )
                else:
                    #   Buffer access is lane-invariant
                    vec_expr = PsVecBroadcast(vc.lanes, expr.clone())

            case PsSubscript(array, index):
                # Check that array expression and indices are lane-invariant
                free_symbols = self._collect_symbols(array).union(
                    *[self._collect_symbols(i) for i in index]
                )
                vec_dependencies = vc.axis_ctr_dependees(free_symbols)
                if vec_dependencies:
                    raise VectorizationError(
                        "Unable to vectorize array subscript depending on vectorized symbols:\n"
                        f"  Offending dependencies:\n"
                        f"    {vec_dependencies}\n"
                        f"  Found in expression:\n"
                        f"{indent(str(expr), '   ')}"
                    )

                vec_expr = PsVecBroadcast(vc.lanes, expr.clone())

            case _:
                raise NotImplementedError(
                    f"Vectorization of {type(expr)} is not implemented"
                )

        vec_expr.dtype = vc.vector_type(scalar_type)
        return vec_expr

    def _index_as_affine(
        self, idx: PsExpression, vc: VectorizationContext
    ) -> Affine | None:
        """Attempt to analyze an index expression as an affine expression of the axis counter."""

        free_symbols = self._collect_symbols(idx)

        #   Check if all symbols except for the axis counter are lane-invariant
        for symb in free_symbols:
            if symb != vc.axis.counter and symb in vc.vectorized_symbols:
                raise VectorizationError(
                    "Unable to rewrite index as affine expression of axis counter: \n"
                    f"   {idx}\n"
                    f"Expression depends on non-lane-invariant symbol {symb}"
                )

        if vc.axis.counter not in free_symbols:
            #   Index is lane-invariant
            return None

        zero = self._factory.parse_index(0)
        one = self._factory.parse_index(1)

        def lane_invariant(expr) -> bool:
            return vc.axis.counter not in self._collect_symbols(expr)

        def collect(subexpr) -> Affine:
            match subexpr:
                case PsSymbolExpr(symb) if symb == vc.axis.counter:
                    return Affine(one, zero)
                case _ if lane_invariant(subexpr):
                    return Affine(zero, subexpr)
                case PsNeg(op):
                    return -collect(op)
                case PsAdd(op1, op2):
                    return collect(op1) + collect(op2)
                case PsSub(op1, op2):
                    return collect(op1) - collect(op2)
                case PsMul(op1, op2) if lane_invariant(op1):
                    return op1 * collect(op2)
                case PsMul(op1, op2) if lane_invariant(op2):
                    return collect(op1) * op2
                case PsDiv(op1, op2) if lane_invariant(op2):
                    return collect(op1) / op2
                case _:
                    raise VectorizationError(
                        "Unable to rewrite index as affine expression of axis counter: \n"
                        f"   {idx}\n"
                        f"Encountered invalid subexpression {subexpr}"
                    )

        return collect(idx)
