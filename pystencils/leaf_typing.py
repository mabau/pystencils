from collections import namedtuple, defaultdict
from typing import List, Union

import numpy as np

import pystencils.integer_functions
import sympy as sp

from pystencils import astnodes as ast, TypedSymbol
from pystencils.bit_masks import flag_cond
from pystencils.field import Field
from pystencils.transformations import NestedScopes
from pystencils.typing import CastFunc, create_type, get_type_of_expression, collate_types
from sympy.codegen import Assignment
from sympy.logic.boolalg import BooleanFunction


class KernelConstraintsCheck: # TODO rename
    # TODO: Logs
    # TODO: specification
    # TODO: split this into checker and leaf typing
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

    def process_assignment(self, assignment: Union[sp.Eq, ast.SympyAssignment, Assignment]) -> ast.SympyAssignment:
        # for checks it is crucial to process rhs before lhs to catch e.g. a = a + 1
        new_rhs = self.process_expression(assignment.rhs)
        new_lhs = self.process_lhs(assignment.lhs)
        return ast.SympyAssignment(new_lhs, new_rhs)


    # Expression
    # 1) ask children if they are cocksure about a type
    # 1b) Postpone clueless children (see 5)
    # cocksure: Children have somewhere type from Field.Access, TypedSymbol, CastFunction or Function^TM
    # clueless: Children without Field.Access,...
    # 1c) none child is cocksure -> do nothing a return None, wait for recall from parent
    # 2) collate_type of children
    # 3) apply collated type on children
    # 4) issue warnings of casts on cocksure children
    # 5a) resume on clueless children with the collated type as default datatype, issue warning
    # 5b) or apply special circumstances

    def process_expression(self, rhs, type_constants=True):  # TODO default_type as parameter
        if isinstance(rhs, Field.Access):
            return rhs
        elif isinstance(rhs, TypedSymbol):
            return rhs
        elif isinstance(rhs, sp.Symbol):
            return TypedSymbol(rhs.name, self._type_for_symbol[rhs.name])
        elif type_constants and isinstance(rhs, np.generic):
            assert False, f'Why do we have a np.generic in rhs???? {rhs}'
            # return CastFunc(rhs, create_type(rhs.dtype))
        elif type_constants and isinstance(rhs, sp.Number):
            return CastFunc(rhs, create_type(self._type_for_symbol['_constant']))
        # Very important that this clause comes before BooleanFunction
        elif isinstance(rhs, sp.Equality):
            if isinstance(rhs.args[1], sp.Number):
                return sp.Equality(
                    self.process_expression(rhs.args[0], type_constants),
                    rhs.args[1])  # TODO: process args[1] as number with a good type
            else:
                return sp.Equality(
                    self.process_expression(rhs.args[0], type_constants),
                    self.process_expression(rhs.args[1], type_constants))
        elif isinstance(rhs, CastFunc):
            return CastFunc(
                self.process_expression(rhs.args[0], type_constants=False),  # TODO: recommend type
                rhs.dtype)
        elif isinstance(rhs, BooleanFunction) or \
                type(rhs) in pystencils.integer_functions.__dict__.values():
            new_args = [self.process_expression(a, type_constants) for a in rhs.args]  # TODO: recommend type
            types_of_expressions = [get_type_of_expression(a) for a in new_args]
            arg_type = collate_types(types_of_expressions, forbid_collation_to_float=True)  # TODO: this must go
            new_args = [a if not hasattr(a, 'dtype') or a.dtype == arg_type
                        else CastFunc(a, arg_type)
                        for a in new_args]
            return rhs.func(*new_args)
        elif isinstance(rhs, flag_cond):  # TODO
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
        elif isinstance(rhs, sp.Pow):
            # don't process exponents -> they should remain integers  # TODO
            return sp.Pow(self.process_expression(rhs.args[0], type_constants), rhs.args[1])
        else:
            new_args = [self.process_expression(arg, type_constants) for arg in rhs.args]
            return rhs.func(*new_args) if new_args else rhs

    @property
    def fields_written(self):
        """
        Return all rhs fields
        """
        return set(k.field for k, v in self.field_writes.items() if len(v))

    def process_lhs(self, lhs: Union[Field.Access, TypedSymbol, sp.Symbol]):
        if not isinstance(lhs, (Field.Access, TypedSymbol)):
            return TypedSymbol(lhs.name, self._type_for_symbol[lhs.name])
        else:
            return lhs
