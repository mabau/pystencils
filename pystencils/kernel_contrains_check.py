from collections import namedtuple, defaultdict
from typing import Union

import sympy as sp
from sympy.codegen import Assignment

from pystencils import astnodes as ast, TypedSymbol
from pystencils.field import Field
from pystencils.transformations import NestedScopes


class KernelConstraintsCheck:
    # TODO: specification
    # TODO: More checks :)
    """Checks if the input to create_kernel is valid.

    Test the following conditions:

    - SSA Form for pure symbols:
        -  Every pure symbol may occur only once as left-hand-side of an assignment
        -  Every pure symbol that is read, may not be written to later
    - Independence / Parallelization condition:
        - a field that is written may only be read at exact the same spatial position

    (Pure symbols are symbols that are not Field.Accesses)
    """
    FieldAndIndex = namedtuple('FieldAndIndex', ['field', 'index'])

    def __init__(self, type_for_symbol, check_independence_condition, check_double_write_condition=True):
        self._type_for_symbol = type_for_symbol

        self.scopes = NestedScopes()
        self.field_writes = defaultdict(set)
        self.fields_read = set()
        self.check_independence_condition = check_independence_condition
        self.check_double_write_condition = check_double_write_condition

    def process_assignment(self, assignment: Union[sp.Eq, ast.SympyAssignment, Assignment]):
        # for checks it is crucial to process rhs before lhs to catch e.g. a = a + 1
        self.process_expression(assignment.rhs)
        self.process_lhs(assignment.lhs)

    def process_expression(self, rhs, type_constants=True):
        self.update_accesses_rhs(rhs)
        if isinstance(rhs, Field.Access):
            self.fields_read.add(rhs.field)
            self.fields_read.update(rhs.indirect_addressing_fields)
        else:
            for arg in rhs.args:
                self.process_expression(arg, type_constants)

    @property
    def fields_written(self):
        """
        Return all rhs fields
        """
        return set(k.field for k, v in self.field_writes.items() if len(v))

    def process_lhs(self, lhs: Union[Field.Access, TypedSymbol, sp.Symbol]):
        assert isinstance(lhs, sp.Symbol)
        self.update_accesses_lhs(lhs)

    def update_accesses_lhs(self, lhs):
        if isinstance(lhs, Field.Access):
            fai = self.FieldAndIndex(lhs.field, lhs.index)
            self.field_writes[fai].add(lhs.offsets)
            if self.check_double_write_condition and len(self.field_writes[fai]) > 1:
                raise ValueError(
                    f"Field {lhs.field.name} is written at two different locations")
        elif isinstance(lhs, sp.Symbol):
            if self.scopes.is_defined_locally(lhs):
                raise ValueError(f"Assignments not in SSA form, multiple assignments to {lhs.name}")
            if lhs in self.scopes.free_parameters:
                raise ValueError(f"Symbol {lhs.name} is written, after it has been read")
            self.scopes.define_symbol(lhs)

    def update_accesses_rhs(self, rhs):
        if isinstance(rhs, Field.Access) and self.check_independence_condition:
            writes = self.field_writes[self.FieldAndIndex(
                rhs.field, rhs.index)]
            for write_offset in writes:
                assert len(writes) == 1
                if write_offset != rhs.offsets:
                    raise ValueError(f"Violation of loop independence condition. Field "
                                     f"{rhs.field} is read at {rhs.offsets} and written at {write_offset}")
            self.fields_read.add(rhs.field)
        elif isinstance(rhs, sp.Symbol):
            self.scopes.access_symbol(rhs)
