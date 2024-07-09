import sympy as sp


class ConditionalFieldAccess(sp.Function):
    """
    :class:`pystencils.Field.Access` that is only executed if a certain condition is met.
    Can be used, for instance, for out-of-bound checks.
    """

    def __new__(cls, field_access, outofbounds_condition, outofbounds_value=0):
        return sp.Function.__new__(cls, field_access, outofbounds_condition, sp.S(outofbounds_value))

    @property
    def access(self):
        return self.args[0]

    @property
    def outofbounds_condition(self):
        return self.args[1]

    @property
    def outofbounds_value(self):
        return self.args[2]

    def __getnewargs__(self):
        return self.access, self.outofbounds_condition, self.outofbounds_value

    def __getnewargs_ex__(self):
        return (self.access, self.outofbounds_condition, self.outofbounds_value), {}


def generic_visit(term, visitor):
    from pystencils import AssignmentCollection, Assignment

    if isinstance(term, AssignmentCollection):
        new_main_assignments = generic_visit(term.main_assignments, visitor)
        new_subexpressions = generic_visit(term.subexpressions, visitor)
        return term.copy(new_main_assignments, new_subexpressions)
    elif isinstance(term, list):
        return [generic_visit(e, visitor) for e in term]
    elif isinstance(term, Assignment):
        return Assignment(term.lhs, generic_visit(term.rhs, visitor))
    elif isinstance(term, sp.Matrix):
        return term.applyfunc(lambda e: generic_visit(e, visitor))
    else:
        return visitor(term)
