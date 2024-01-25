from __future__ import annotations

from collections import namedtuple, defaultdict
from typing import Any, Sequence
from itertools import chain

import sympy as sp

from .context import KernelCreationContext

from ...field import Field
from ...assignment import Assignment
from ...simp import AssignmentCollection
from ...transformations import NestedScopes

from ..exceptions import PsInternalCompilerError


class KernelConstraintsError(Exception):
    pass


class KernelAnalysis:
    """General analysis pass over a kernel expressed using the SymPy frontend.

    The kernel analysis fulfills two tasks. It checks the SymPy input for consistency,
    and populates the context with required knowledge about the kernel.

    A `KernelAnalysis` object may be called at most once.

    Consistency and Constraints
    ---------------------------

    The following checks are performed:

     - **SSA Form:** The given assignments must be in single-assignment form; each symbol must be written at most once.
     - **Independence of Accesses:** To avoid loop-carried dependencies, each field may be written at most once at
       each index, and if a field is written at some location with index `i`, it may only be read with index `i` in
       the same location.
     - **Independence of Writes:** A weaker requirement than access independence; each field may only be written once
       at each index.

    Knowledge Collection
    --------------------

    The following knowledge is collected into the context:
     - The set of fields accessed in the kernel
    """

    FieldAndIndex = namedtuple("FieldAndIndex", ["field", "index"])

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

        self._check_access_independence = False
        self._check_double_writes = (
            False  # TODO: Access independence check implies double writes check
        )

        #   Map pairs of fields and indices to offsets
        self._field_writes: dict[KernelAnalysis.FieldAndIndex, set[Any]] = defaultdict(
            set
        )

        self._fields_written: set[Field] = set()
        self._fields_read: set[Field] = set()

        self._scopes = NestedScopes()

        self._called = False

    def __call__(self, obj: AssignmentCollection | Sequence[Assignment] | Assignment):
        if self._called:
            raise PsInternalCompilerError("KernelAnalysis called twice!")

        self._called = True
        self._visit(obj)

        for field in chain(self._fields_written, self._fields_read):
            self._ctx.add_field(field)

    def _visit(self, obj: Any):
        match obj:
            case AssignmentCollection(main_asms, subexps):
                self._visit(subexps)
                self._visit(main_asms)

            case [*asms]:  # lists and tuples are unpacked
                for asm in asms:
                    self._visit(asm)

            case Assignment():
                self._handle_rhs(obj.rhs)
                self._handle_lhs(obj.lhs)

            case unknown:
                raise KernelConstraintsError(
                    f"Don't know how to interpret {unknown} in a kernel."
                )

    def _handle_lhs(self, lhs: sp.Basic):
        if not isinstance(lhs, sp.Symbol):
            raise KernelConstraintsError(
                f"Invalid expression on assignment left-hand side: {lhs}"
            )

        match lhs:
            case Field.Access(field, offsets, index):
                self._fields_written.add(field)
                self._fields_read.update(lhs.indirect_addressing_fields)

                fai = self.FieldAndIndex(field, index)
                if self._check_double_writes and offsets in self._field_writes[fai]:
                    raise KernelConstraintsError(
                        f"Field {field.name} is written twice at the same location"
                    )

                self._field_writes[fai].add(offsets)

                if self._check_double_writes and len(self._field_writes[fai]) > 1:
                    raise KernelConstraintsError(
                        f"Field {field.name} is written at two different locations"
                    )

            case sp.Symbol():
                if self._scopes.is_defined_locally(lhs):
                    raise KernelConstraintsError(
                        f"Assignments not in SSA form, multiple assignments to {lhs.name}"
                    )
                if lhs in self._scopes.free_parameters:
                    raise KernelConstraintsError(
                        f"Symbol {lhs.name} is written, after it has been read"
                    )
                self._scopes.define_symbol(lhs)

    def _handle_rhs(self, rhs: sp.Basic):
        def rec(expr: sp.Basic):
            match expr:
                case Field.Access(field, offsets, index):
                    self._fields_read.add(field)
                    self._fields_read.update(expr.indirect_addressing_fields)
                    #   TODO: Should we recurse into the arguments of the field access?

                    if self._check_access_independence:
                        writes = self._field_writes[
                            KernelAnalysis.FieldAndIndex(field, index)
                        ]
                        assert len(writes) <= 1
                        for write_offset in writes:
                            if write_offset != offsets:
                                raise KernelConstraintsError(
                                    f"Violation of loop independence condition. Field "
                                    f"{field} is read at {offsets} and written at {write_offset}"
                                )
                case sp.Symbol():
                    self._scopes.access_symbol(expr)

            for arg in expr.args:
                rec(arg)

        rec(rhs)
