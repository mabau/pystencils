from collections import namedtuple, defaultdict

import numpy as np

import pystencils.integer_functions
import sympy as sp
from pystencils import astnodes as ast, TypedSymbol
from pystencils.bit_masks import flag_cond
from pystencils.field import AbstractField
from pystencils.transformations import NestedScopes
from pystencils.typing import CastFunc, create_type, get_type_of_expression, collate_types
from sympy.logic.boolalg import BooleanFunction


class KernelConstraintsCheck:
    # TODO: Logs
    # TODO: specification
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
        self._field_writes = defaultdict(set)
        self.fields_read = set()
        self.check_independence_condition = check_independence_condition
        self.check_double_write_condition = check_double_write_condition

    def process_assignment(self, assignment):
        # for checks it is crucial to process rhs before lhs to catch e.g. a = a + 1
        new_rhs = self.process_expression(assignment.rhs)
        new_lhs = self._process_lhs(assignment.lhs)
        return ast.SympyAssignment(new_lhs, new_rhs)

    def process_expression(self, rhs, type_constants=True):

        self._update_accesses_rhs(rhs)
        if isinstance(rhs, AbstractField.AbstractAccess):
            self.fields_read.add(rhs.field)
            self.fields_read.update(rhs.indirect_addressing_fields)
            return rhs
        # TODO remove this
        #elif isinstance(rhs, ImaginaryUnit):
        #    return TypedImaginaryUnit(create_type(self._type_for_symbol['_complex_type']))
        elif isinstance(rhs, TypedSymbol):
            return rhs
        elif isinstance(rhs, sp.Symbol):
            return TypedSymbol(rhs.name, self._type_for_symbol[rhs.name])
        elif type_constants and isinstance(rhs, np.generic):
            return CastFunc(rhs, create_type(rhs.dtype))
        elif type_constants and isinstance(rhs, sp.Number):
            return CastFunc(rhs, create_type(self._type_for_symbol['_constant']))
        # Very important that this clause comes before BooleanFunction
        elif isinstance(rhs, sp.Equality):
            if isinstance(rhs.args[1], sp.Number):
                return sp.Equality(
                    self.process_expression(rhs.args[0], type_constants),
                    rhs.args[1])
            else:
                return sp.Equality(
                    self.process_expression(rhs.args[0], type_constants),
                    self.process_expression(rhs.args[1], type_constants))
        elif isinstance(rhs, CastFunc):
            return CastFunc(
                self.process_expression(rhs.args[0], type_constants=False),
                rhs.dtype)
        elif isinstance(rhs, BooleanFunction) or \
                type(rhs) in pystencils.integer_functions.__dict__.values():
            new_args = [self.process_expression(a, type_constants) for a in rhs.args]
            types_of_expressions = [get_type_of_expression(a) for a in new_args]
            arg_type = collate_types(types_of_expressions, forbid_collation_to_float=True)
            new_args = [a if not hasattr(a, 'dtype') or a.dtype == arg_type
                        else CastFunc(a, arg_type)
                        for a in new_args]
            return rhs.func(*new_args)
        elif isinstance(rhs, flag_cond):
            #   do not process the arguments to the bit shift - they must remain integers
            processed_args = (self.process_expression(a) for a in rhs.args[2:])
            return flag_cond(rhs.args[0], rhs.args[1], *processed_args)
        elif isinstance(rhs, sp.Mul):
            new_args = [
                self.process_expression(arg, type_constants)
                if arg not in (-1, 1) else arg for arg in rhs.args
            ]
            return rhs.func(*new_args) if new_args else rhs
        elif isinstance(rhs, sp.Indexed):
            return rhs
        else:
            if isinstance(rhs, sp.Pow):
                # don't process exponents -> they should remain integers
                return sp.Pow(
                    self.process_expression(rhs.args[0], type_constants),
                    rhs.args[1])
            else:
                new_args = [
                    self.process_expression(arg, type_constants)
                    for arg in rhs.args
                ]
                return rhs.func(*new_args) if new_args else rhs

    @property
    def fields_written(self):
        return set(k.field for k, v in self._field_writes.items() if len(v))

    def _process_lhs(self, lhs):
        assert isinstance(lhs, sp.Symbol)
        self._update_accesses_lhs(lhs)
        if not isinstance(lhs, (AbstractField.AbstractAccess, TypedSymbol)):
            return TypedSymbol(lhs.name, self._type_for_symbol[lhs.name])
        else:
            return lhs

    def _update_accesses_lhs(self, lhs):
        if isinstance(lhs, AbstractField.AbstractAccess):
            fai = self.FieldAndIndex(lhs.field, lhs.index)
            self._field_writes[fai].add(lhs.offsets)
            if self.check_double_write_condition and len(self._field_writes[fai]) > 1:
                raise ValueError(
                    f"Field {lhs.field.name} is written at two different locations")
        elif isinstance(lhs, sp.Symbol):
            if self.scopes.is_defined_locally(lhs):
                raise ValueError(f"Assignments not in SSA form, multiple assignments to {lhs.name}")
            if lhs in self.scopes.free_parameters:
                raise ValueError(f"Symbol {lhs.name} is written, after it has been read")
            self.scopes.define_symbol(lhs)

    def _update_accesses_rhs(self, rhs):
        if isinstance(rhs, AbstractField.AbstractAccess) and self.check_independence_condition:
            writes = self._field_writes[self.FieldAndIndex(
                rhs.field, rhs.index)]
            for write_offset in writes:
                assert len(writes) == 1
                if write_offset != rhs.offsets:
                    raise ValueError("Violation of loop independence condition. Field "
                                     "{} is read at {} and written at {}".format(rhs.field, rhs.offsets, write_offset))
            self.fields_read.add(rhs.field)
        elif isinstance(rhs, sp.Symbol):
            self.scopes.access_symbol(rhs)