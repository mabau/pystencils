from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

from pymbolic.mapper.dependency import DependencyMapper

from .nodes import PsAstNode, PsBlock, failing_cast

from ..constraints import PsKernelConstraint
from ..typed_expressions import PsTypedVariable
from ..arrays import PsLinearizedArray, PsArrayBasePointer, PsArrayAssocVar
from ..jit import JitBase, no_jit
from ..exceptions import PsInternalCompilerError

from ...enums import Target


@dataclass
class PsKernelParametersSpec:
    """Specification of a kernel function's parameters.

    Contains:
        - Verbatim parameter list, a list of `PsTypedVariables`
        - List of Arrays used in the kernel, in canonical order
        - A set of constraints on the kernel parameters, used to e.g. express relations of array
          shapes, alignment properties, ...
    """

    params: tuple[PsTypedVariable, ...]
    arrays: tuple[PsLinearizedArray, ...]
    constraints: tuple[PsKernelConstraint, ...]

    def params_for_array(self, arr: PsLinearizedArray):
        def pred(p: PsTypedVariable):
            return isinstance(p, PsArrayAssocVar) and p.array == arr

        return tuple(filter(pred, self.params))

    def __post_init__(self):
        dep_mapper = DependencyMapper(False, False, False, False)

        #   Check constraints
        for constraint in self.constraints:
            variables: set[PsTypedVariable] = dep_mapper(constraint.condition)
            for var in variables:
                if isinstance(var, PsArrayAssocVar):
                    if var.array in self.arrays:
                        continue

                elif var in self.params:
                    continue

                raise PsInternalCompilerError(
                    "Constrained parameter was neither contained in kernel parameter list "
                    "nor associated with a kernel array.\n"
                    f"    Parameter: {var}\n"
                    f"    Constraint: {constraint.condition}"
                )


class PsKernelFunction(PsAstNode):
    """A pystencils kernel function.

    Objects of this class represent a full pystencils kernel and should provide all information required for
    export, compilation, and inclusion of the kernel into a runtime system.
    """

    __match_args__ = ("body",)

    def __init__(
        self, body: PsBlock, target: Target, name: str = "kernel", jit: JitBase = no_jit
    ):
        self._body: PsBlock = body
        self._target = target
        self._name = name
        self._jit = jit

        self._constraints: list[PsKernelConstraint] = []

    @property
    def target(self) -> Target:
        """See pystencils.Target"""
        return self._target

    @property
    def body(self) -> PsBlock:
        return self._body

    @body.setter
    def body(self, body: PsBlock):
        self._body = body

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def function_name(self) -> str:
        """For backward compatibility."""
        return self._name

    @property
    def instruction_set(self) -> str | None:
        """For backward compatibility"""
        return None

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._body,)

    def set_child(self, idx: int, c: PsAstNode):
        if idx not in (0, -1):
            raise IndexError(f"Child index out of bounds: {idx}")
        self._body = failing_cast(PsBlock, c)

    def add_constraints(self, *constraints: PsKernelConstraint):
        self._constraints += constraints

    def get_parameters(self) -> PsKernelParametersSpec:
        """Collect the list of parameters to this function.

        This function performs a full traversal of the AST.
        To improve performance, make sure to cache the result if necessary.
        """
        from .collectors import collect_undefined_variables

        params_set = collect_undefined_variables(self)
        params_list = sorted(params_set, key=lambda p: p.name)

        arrays = set(p.array for p in params_list if isinstance(p, PsArrayBasePointer))
        return PsKernelParametersSpec(
            tuple(params_list), tuple(arrays), tuple(self._constraints)
        )

    def get_required_headers(self) -> set[str]:
        #   To Do: Headers from target/instruction set/...
        from .collectors import collect_required_headers

        return collect_required_headers(self)

    def compile(self) -> Callable[..., None]:
        return self._jit.compile(self)
