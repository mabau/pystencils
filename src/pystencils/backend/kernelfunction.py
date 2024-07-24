from __future__ import annotations

from warnings import warn
from abc import ABC
from typing import Callable, Sequence, Iterable, Any, TYPE_CHECKING

from .._deprecation import _deprecated

from .ast.structural import PsBlock
from .ast.analysis import collect_required_headers, collect_undefined_symbols
from .arrays import PsArrayShapeSymbol, PsArrayStrideSymbol, PsArrayBasePointer
from .symbols import PsSymbol
from .kernelcreation.context import KernelCreationContext
from .platforms import Platform, GpuThreadsRange

from .constraints import KernelParamsConstraint
from ..types import PsType

from ..enums import Target
from ..field import Field
from ..sympyextensions import TypedSymbol

if TYPE_CHECKING:
    from .jit import JitBase


class KernelParameter:
    __match_args__ = ("name", "dtype")

    def __init__(self, name: str, dtype: PsType):
        self._name = name
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    def _hashable_contents(self):
        return (self._name, self._dtype)

    def __hash__(self) -> int:
        return hash(self._hashable_contents())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KernelParameter):
            return False

        return (
            type(self) is type(other)
            and self._hashable_contents() == other._hashable_contents()
        )

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name = {self._name}, dtype = {self._dtype})"

    @property
    def symbol(self) -> TypedSymbol:
        return TypedSymbol(self.name, self.dtype)

    @property
    def is_field_parameter(self) -> bool:
        warn(
            "`is_field_parameter` is deprecated and will be removed in a future version of pystencils. "
            "Use `isinstance(param, FieldParameter)` instead.",
            DeprecationWarning,
        )
        return isinstance(self, FieldParameter)

    @property
    def is_field_pointer(self) -> bool:
        warn(
            "`is_field_pointer` is deprecated and will be removed in a future version of pystencils. "
            "Use `isinstance(param, FieldPointerParam)` instead.",
            DeprecationWarning,
        )
        return isinstance(self, FieldPointerParam)

    @property
    def is_field_stride(self) -> bool:
        warn(
            "`is_field_stride` is deprecated and will be removed in a future version of pystencils. "
            "Use `isinstance(param, FieldStrideParam)` instead.",
            DeprecationWarning,
        )
        return isinstance(self, FieldStrideParam)

    @property
    def is_field_shape(self) -> bool:
        warn(
            "`is_field_shape` is deprecated and will be removed in a future version of pystencils. "
            "Use `isinstance(param, FieldShapeParam)` instead.",
            DeprecationWarning,
        )
        return isinstance(self, FieldShapeParam)


class FieldParameter(KernelParameter, ABC):
    __match_args__ = KernelParameter.__match_args__ + ("field",)

    def __init__(self, name: str, dtype: PsType, field: Field):
        super().__init__(name, dtype)
        self._field = field

    @property
    def field(self):
        return self._field

    @property
    def fields(self):
        warn(
            "`fields` is deprecated and will be removed in a future version of pystencils. "
            "In pystencils >= 2.0, field parameters are only associated with a single field."
            "Use the `field` property instead.",
            DeprecationWarning,
        )
        return [self._field]

    @property
    def field_name(self) -> str:
        warn(
            "`field_name` is deprecated and will be removed in a future version of pystencils. "
            "Use `field.name` instead.",
            DeprecationWarning,
        )
        return self._field.name

    def _hashable_contents(self):
        return super()._hashable_contents() + (self._field,)


class FieldShapeParam(FieldParameter):
    __match_args__ = FieldParameter.__match_args__ + ("coordinate",)

    def __init__(self, name: str, dtype: PsType, field: Field, coordinate: int):
        super().__init__(name, dtype, field)
        self._coordinate = coordinate

    @property
    def coordinate(self):
        return self._coordinate

    def _hashable_contents(self):
        return super()._hashable_contents() + (self._coordinate,)


class FieldStrideParam(FieldParameter):
    __match_args__ = FieldParameter.__match_args__ + ("coordinate",)

    def __init__(self, name: str, dtype: PsType, field: Field, coordinate: int):
        super().__init__(name, dtype, field)
        self._coordinate = coordinate

    @property
    def coordinate(self):
        return self._coordinate

    def _hashable_contents(self):
        return super()._hashable_contents() + (self._coordinate,)


class FieldPointerParam(FieldParameter):
    def __init__(self, name: str, dtype: PsType, field: Field):
        super().__init__(name, dtype, field)


class KernelFunction:
    """A pystencils kernel function.

    The kernel function is the final result of the translation process.
    It is immutable, and its AST should not be altered any more, either, as this
    might invalidate information about the kernel already stored in the `KernelFunction` object.
    """

    def __init__(
        self,
        body: PsBlock,
        target: Target,
        name: str,
        parameters: Sequence[KernelParameter],
        required_headers: set[str],
        constraints: Sequence[KernelParamsConstraint],
        jit: JitBase,
    ):
        self._body: PsBlock = body
        self._target = target
        self._name = name
        self._params = tuple(parameters)
        self._required_headers = required_headers
        self._constraints = tuple(constraints)
        self._jit = jit
        self._metadata: dict[str, Any] = dict()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def body(self) -> PsBlock:
        return self._body

    @property
    def target(self) -> Target:
        return self._target

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n

    @property
    def function_name(self) -> str:
        _deprecated("function_name", "name")
        return self._name

    @function_name.setter
    def function_name(self, n: str):
        _deprecated("function_name", "name")
        self._name = n

    @property
    def parameters(self) -> tuple[KernelParameter, ...]:
        return self._params

    def get_parameters(self) -> tuple[KernelParameter, ...]:
        _deprecated("KernelFunction.get_parameters", "KernelFunction.parameters")
        return self.parameters

    def get_fields(self) -> set[Field]:
        return set(p.field for p in self._params if isinstance(p, FieldParameter))

    @property
    def fields_accessed(self) -> set[Field]:
        warn(
            "`fields_accessed` is deprecated and will be removed in a future version of pystencils. "
            "Use `get_fields` instead.",
            DeprecationWarning,
        )
        return self.get_fields()

    @property
    def required_headers(self) -> set[str]:
        return self._required_headers

    @property
    def constraints(self) -> tuple[KernelParamsConstraint, ...]:
        return self._constraints

    def compile(self) -> Callable[..., None]:
        return self._jit.compile(self)


def create_cpu_kernel_function(
    ctx: KernelCreationContext,
    platform: Platform,
    body: PsBlock,
    function_name: str,
    target_spec: Target,
    jit: JitBase,
):
    undef_symbols = collect_undefined_symbols(body)

    params = _get_function_params(ctx, undef_symbols)
    req_headers = _get_headers(ctx, platform, body)

    kfunc = KernelFunction(
        body, target_spec, function_name, params, req_headers, ctx.constraints, jit
    )
    kfunc.metadata.update(ctx.metadata)
    return kfunc


class GpuKernelFunction(KernelFunction):
    def __init__(
        self,
        body: PsBlock,
        threads_range: GpuThreadsRange,
        target: Target,
        name: str,
        parameters: Sequence[KernelParameter],
        required_headers: set[str],
        constraints: Sequence[KernelParamsConstraint],
        jit: JitBase,
    ):
        super().__init__(
            body, target, name, parameters, required_headers, constraints, jit
        )
        self._threads_range = threads_range

    @property
    def threads_range(self) -> GpuThreadsRange:
        return self._threads_range


def create_gpu_kernel_function(
    ctx: KernelCreationContext,
    platform: Platform,
    body: PsBlock,
    threads_range: GpuThreadsRange,
    function_name: str,
    target_spec: Target,
    jit: JitBase,
):
    undef_symbols = collect_undefined_symbols(body)
    for threads in threads_range.num_work_items:
        undef_symbols |= collect_undefined_symbols(threads)

    params = _get_function_params(ctx, undef_symbols)
    req_headers = _get_headers(ctx, platform, body)

    kfunc = GpuKernelFunction(
        body,
        threads_range,
        target_spec,
        function_name,
        params,
        req_headers,
        ctx.constraints,
        jit,
    )
    kfunc.metadata.update(ctx.metadata)
    return kfunc


def _get_function_params(ctx: KernelCreationContext, symbols: Iterable[PsSymbol]):
    params: list[KernelParameter] = []
    for symb in symbols:
        match symb:
            case PsArrayShapeSymbol(name, _, arr, coord):
                field = ctx.find_field(arr.name)
                params.append(FieldShapeParam(name, symb.get_dtype(), field, coord))
            case PsArrayStrideSymbol(name, _, arr, coord):
                field = ctx.find_field(arr.name)
                params.append(FieldStrideParam(name, symb.get_dtype(), field, coord))
            case PsArrayBasePointer(name, _, arr):
                field = ctx.find_field(arr.name)
                params.append(FieldPointerParam(name, symb.get_dtype(), field))
            case PsSymbol(name, _):
                params.append(KernelParameter(name, symb.get_dtype()))

    params.sort(key=lambda p: p.name)
    return params


def _get_headers(ctx: KernelCreationContext, platform: Platform, body: PsBlock):
    req_headers = collect_required_headers(body)
    req_headers |= platform.required_headers
    req_headers |= ctx.required_headers
    return req_headers
