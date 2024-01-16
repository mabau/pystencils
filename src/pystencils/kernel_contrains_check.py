from collections import namedtuple, defaultdict
from typing import Union

import sympy as sp
from sympy.codegen import Assignment

from pystencils.simp import AssignmentCollection
from pystencils import astnodes as ast, TypedSymbol
from pystencils.field import Field
from pystencils.node_collection import NodeCollection
from pystencils.transformations import NestedScopes

# TODO use this in Constraint Checker
accepted_functions = [
    sp.Pow,
    sp.sqrt,
    sp.log,
    # TODO trigonometric functions (and whatever tests will fail)
]


class KernelConstraintsCheck:
    # TODO: proper specification
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

    def __init__(self, check_independence_condition=True, check_double_write_condition=True):
        self.scopes = NestedScopes()
        self.field_writes = defaultdict(set)
        self.fields_read = set()
        self.check_independence_condition = check_independence_condition
        self.check_double_write_condition = check_double_write_condition

    def visit(self, obj):
        if isinstance(obj, (AssignmentCollection, NodeCollection)):
            [self.visit(e) for e in obj.all_assignments]
        elif isinstance(obj, list) or isinstance(obj, tuple):
            [self.visit(e) for e in obj]
        elif isinstance(obj, (sp.Eq, ast.SympyAssignment, Assignment)):
            self.process_assignment(obj)
        elif isinstance(obj, ast.Conditional):
            self.scopes.push()
            # Disable double write check inside conditionals
            # would be triggered by e.g. in-kernel boundaries
            old_double_write = self.check_double_write_condition
            old_independence_condition = self.check_independence_condition
            self.check_double_write_condition = False
            self.check_independence_condition = False
            if obj.false_block:
                self.visit(obj.false_block)
            self.process_expression(obj.condition_expr)
            self.process_expression(obj.true_block)
            self.check_double_write_condition = old_double_write
            self.check_independence_condition = old_independence_condition
            self.scopes.pop()
        elif isinstance(obj, ast.Block):
            self.scopes.push()
            [self.visit(e) for e in obj.args]
            self.scopes.pop()
        elif isinstance(obj, ast.Node) and not isinstance(obj, ast.LoopOverCoordinate):
            pass
        else:
            raise ValueError(f'Invalid object in kernel {type(obj)}')

    def process_assignment(self, assignment: Union[sp.Eq, ast.SympyAssignment, Assignment]):
        # for checks it is crucial to process rhs before lhs to catch e.g. a = a + 1
        self.process_expression(assignment.rhs)
        self.process_lhs(assignment.lhs)

    def process_expression(self, rhs):
        # TODO constraint for accepted functions, see TODO above
        self.update_accesses_rhs(rhs)
        if isinstance(rhs, Field.Access):
            self.fields_read.add(rhs.field)
            self.fields_read.update(rhs.indirect_addressing_fields)
        else:
            for arg in rhs.args:
                self.process_expression(arg)

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
            if self.check_double_write_condition and lhs.offsets in self.field_writes[fai]:
                raise ValueError(f"Field {lhs.field.name} is written twice at the same location")

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
