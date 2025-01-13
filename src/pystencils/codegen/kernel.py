from __future__ import annotations

from warnings import warn
from typing import Callable, Sequence, Any, TYPE_CHECKING
from itertools import chain

from .target import Target
from .parameters import Parameter
from ..backend.ast.structural import PsBlock
from ..backend.ast.expressions import PsExpression
from ..field import Field

from .._deprecation import _deprecated

if TYPE_CHECKING:
    from ..jit import JitBase


class Kernel:
    """A pystencils kernel.

    The kernel object is the final result of the translation process.
    It is immutable, and its AST should not be altered any more, either, as this
    might invalidate information about the kernel already stored in the kernel object.
    """

    def __init__(
        self,
        body: PsBlock,
        target: Target,
        name: str,
        parameters: Sequence[Parameter],
        required_headers: set[str],
        jit: JitBase,
    ):
        self._body: PsBlock = body
        self._target = target
        self._name = name
        self._params = tuple(parameters)
        self._required_headers = required_headers
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
    def function_name(self) -> str:  # pragma: no cover
        _deprecated("function_name", "name")
        return self._name

    @function_name.setter
    def function_name(self, n: str):  # pragma: no cover
        _deprecated("function_name", "name")
        self._name = n

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return self._params

    def get_parameters(self) -> tuple[Parameter, ...]:  # pragma: no cover
        _deprecated("Kernel.get_parameters", "Kernel.parameters")
        return self.parameters

    def get_fields(self) -> set[Field]:
        return set(chain.from_iterable(p.fields for p in self._params))

    @property
    def fields_accessed(self) -> set[Field]:  # pragma: no cover
        warn(
            "`fields_accessed` is deprecated and will be removed in a future version of pystencils. "
            "Use `get_fields` instead.",
            DeprecationWarning,
        )
        return self.get_fields()

    @property
    def required_headers(self) -> set[str]:
        return self._required_headers

    def get_c_code(self) -> str:
        from ..backend.emission import CAstPrinter

        printer = CAstPrinter()
        return printer(self)

    def get_ir_code(self) -> str:
        from ..backend.emission import IRAstPrinter

        printer = IRAstPrinter()
        return printer(self)

    def compile(self) -> Callable[..., None]:
        """Invoke the underlying just-in-time compiler to obtain the kernel as an executable Python function."""
        return self._jit.compile(self)


class GpuKernel(Kernel):
    """Internal representation of a kernel function targeted at CUDA GPUs."""

    def __init__(
        self,
        body: PsBlock,
        threads_range: GpuThreadsRange | None,
        target: Target,
        name: str,
        parameters: Sequence[Parameter],
        required_headers: set[str],
        jit: JitBase,
    ):
        super().__init__(body, target, name, parameters, required_headers, jit)
        self._threads_range = threads_range

    @property
    def threads_range(self) -> GpuThreadsRange | None:
        """Object exposing the total size of the launch grid this kernel expects to be executed with."""
        return self._threads_range


class GpuThreadsRange:
    """Number of threads required by a GPU kernel, in order (x, y, z)."""

    def __init__(
        self,
        num_work_items: Sequence[PsExpression],
    ):
        self._dim = len(num_work_items)
        self._num_work_items = tuple(num_work_items)

    # @property
    # def grid_size(self) -> tuple[PsExpression, ...]:
    #     return self._grid_size

    # @property
    # def block_size(self) -> tuple[PsExpression, ...]:
    #     return self._block_size

    @property
    def num_work_items(self) -> tuple[PsExpression, ...]:
        """Number of work items in (x, y, z)-order."""
        return self._num_work_items

    @property
    def dim(self) -> int:
        return self._dim

    def __str__(self) -> str:
        rep = "GpuThreadsRange { "
        rep += "; ".join(f"{x}: {w}" for x, w in zip("xyz", self._num_work_items))
        rep += " }"
        return rep

    def _repr_html_(self) -> str:
        return str(self)
