from __future__ import annotations

from warnings import warn
from typing import Callable, Sequence, Iterable, Any, TYPE_CHECKING
from itertools import chain

from .._deprecation import _deprecated

from .ast.structural import PsBlock
from .ast.analysis import collect_required_headers, collect_undefined_symbols
from .memory import PsSymbol
from .properties import (
    PsSymbolProperty,
    _FieldProperty,
    FieldShape,
    FieldStride,
    FieldBasePtr,
)
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
    """Parameter to a `KernelFunction`."""

    __match_args__ = ("name", "dtype", "properties")

    def __init__(
        self, name: str, dtype: PsType, properties: Iterable[PsSymbolProperty] = ()
    ):
        self._name = name
        self._dtype = dtype
        self._properties: frozenset[PsSymbolProperty] = (
            frozenset(properties) if properties is not None else frozenset()
        )
        self._fields: tuple[Field, ...] = tuple(
            sorted(
                set(
                    p.field  # type: ignore
                    for p in filter(
                        lambda p: isinstance(p, _FieldProperty), self._properties
                    )
                ),
                key=lambda f: f.name
            )
        )

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    def _hashable_contents(self):
        return (self._name, self._dtype, self._properties)

    #   TODO: Need?
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
    def fields(self) -> Sequence[Field]:
        """Set of fields associated with this parameter."""
        return self._fields

    def get_properties(
        self, prop_type: type[PsSymbolProperty] | tuple[type[PsSymbolProperty], ...]
    ) -> set[PsSymbolProperty]:
        """Retrieve all properties of the given type(s) attached to this parameter"""
        return set(filter(lambda p: isinstance(p, prop_type), self._properties))

    @property
    def properties(self) -> frozenset[PsSymbolProperty]:
        return self._properties

    @property
    def is_field_parameter(self) -> bool:
        return bool(self._fields)

    #   Deprecated legacy properties
    #   These are kept mostly for the legacy waLBerla code generation system

    @property
    def is_field_pointer(self) -> bool:
        warn(
            "`is_field_pointer` is deprecated and will be removed in a future version of pystencils. "
            "Use `param.get_properties(FieldBasePtr)` instead.",
            DeprecationWarning,
        )
        return bool(self.get_properties(FieldBasePtr))

    @property
    def is_field_stride(self) -> bool:
        warn(
            "`is_field_stride` is deprecated and will be removed in a future version of pystencils. "
            "Use `param.get_properties(FieldStride)` instead.",
            DeprecationWarning,
        )
        return bool(self.get_properties(FieldStride))

    @property
    def is_field_shape(self) -> bool:
        warn(
            "`is_field_shape` is deprecated and will be removed in a future version of pystencils. "
            "Use `param.get_properties(FieldShape)` instead.",
            DeprecationWarning,
        )
        return bool(self.get_properties(FieldShape))

    @property
    def field_name(self) -> str:
        warn(
            "`field_name` is deprecated and will be removed in a future version of pystencils. "
            "Use `param.fields[0].name` instead.",
            DeprecationWarning,
        )
        return self._fields[0].name


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
        return set(chain.from_iterable(p.fields for p in self._params))

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
        params.append(KernelParameter(symb.name, symb.get_dtype(), props))

    params.sort(key=lambda p: p.name)
    return params


def _get_headers(ctx: KernelCreationContext, platform: Platform, body: PsBlock):
    req_headers = collect_required_headers(body)
    req_headers |= platform.required_headers
    req_headers |= ctx.required_headers
    return req_headers
