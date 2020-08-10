import pytest
import sympy as sp
import pystencils as ps

from pystencils import Assignment, AssignmentCollection
from pystencils.astnodes import Conditional
from pystencils.simp.assignment_collection import SymbolGen

a, b, c = sp.symbols("a b c")
x, y, z, t = sp.symbols("x y z t")
symbol_gen = SymbolGen("a")
f = ps.fields("f(2) : [2D]")
d = ps.fields("d(2) : [2D]")


def test_assignment_collection():
    ac = AssignmentCollection([Assignment(z, x + y)],
                              [], subexpression_symbol_generator=symbol_gen)

    lhs = ac.add_subexpression(t)
    assert lhs == sp.Symbol("a_0")
    ac.subexpressions.append(Assignment(t, 3))
    ac.topological_sort(sort_main_assignments=False, sort_subexpressions=True)
    assert ac.subexpressions[0].lhs == t

    assert ac.new_with_inserted_subexpression(sp.Symbol("not_defined")) == ac
    ac_inserted = ac.new_with_inserted_subexpression(t)
    ac_inserted2 = ac.new_without_subexpressions({lhs})
    assert all(a == b for a, b in zip(ac_inserted.all_assignments, ac_inserted2.all_assignments))

    print(ac_inserted)
    assert ac_inserted.subexpressions[0] == Assignment(lhs, 3)

    assert 'a_0' in str(ac_inserted)
    assert '<table' in ac_inserted._repr_html_()


def test_free_and_defined_symbols():
    ac = AssignmentCollection([Assignment(z, x + y), Conditional(t > 0, Assignment(a, b+1), Assignment(a, b+2))],
                              [], subexpression_symbol_generator=symbol_gen)

    print(ac)
    print(ac.__repr__)


def test_vector_assignments():
    """From #17 (https://i10git.cs.fau.de/pycodegen/pystencils/issues/17)"""
    assignments = ps.Assignment(sp.Matrix([a, b, c]), sp.Matrix([1, 2, 3]))
    print(assignments)


def test_wrong_vector_assignments():
    """From #17 (https://i10git.cs.fau.de/pycodegen/pystencils/issues/17)"""
    with pytest.raises(AssertionError,
                       match=r'Matrix(.*) and Matrix(.*) must have same length when performing vector assignment!'):
        ps.Assignment(sp.Matrix([a, b]), sp.Matrix([1, 2, 3]))


def test_vector_assignment_collection():
    """From #17 (https://i10git.cs.fau.de/pycodegen/pystencils/issues/17)"""

    y_m, x_m = sp.Matrix([a, b, c]), sp.Matrix([1, 2, 3])
    assignments = ps.AssignmentCollection({y_m: x_m})
    print(assignments)

    assignments = ps.AssignmentCollection([ps.Assignment(y_m, x_m)])
    print(assignments)


def test_new_with_substitutions():
    a1 = ps.Assignment(f[0, 0](0), a * b)
    a2 = ps.Assignment(f[0, 0](1), b * c)

    ac = ps.AssignmentCollection([a1, a2], subexpressions=[])
    subs_dict = {f[0, 0](0): d[0, 0](0), f[0, 0](1): d[0, 0](1)}
    subs_ac = ac.new_with_substitutions(subs_dict,
                                        add_substitutions_as_subexpressions=False,
                                        substitute_on_lhs=True,
                                        sort_topologically=True)

    assert subs_ac.main_assignments[0].lhs == d[0, 0](0)
    assert subs_ac.main_assignments[1].lhs == d[0, 0](1)

    subs_ac = ac.new_with_substitutions(subs_dict,
                                        add_substitutions_as_subexpressions=False,
                                        substitute_on_lhs=False,
                                        sort_topologically=True)

    assert subs_ac.main_assignments[0].lhs == f[0, 0](0)
    assert subs_ac.main_assignments[1].lhs == f[0, 0](1)

    subs_dict = {a * b: sp.symbols('xi')}
    subs_ac = ac.new_with_substitutions(subs_dict,
                                        add_substitutions_as_subexpressions=False,
                                        substitute_on_lhs=False,
                                        sort_topologically=True)

    assert subs_ac.main_assignments[0].rhs == sp.symbols('xi')
    assert len(subs_ac.subexpressions) == 0

    subs_ac = ac.new_with_substitutions(subs_dict,
                                        add_substitutions_as_subexpressions=True,
                                        substitute_on_lhs=False,
                                        sort_topologically=True)

    assert subs_ac.main_assignments[0].rhs == sp.symbols('xi')
    assert len(subs_ac.subexpressions) == 1
    assert subs_ac.subexpressions[0].lhs == sp.symbols('xi')


def test_copy():
    a1 = ps.Assignment(f[0, 0](0), a * b)
    a2 = ps.Assignment(f[0, 0](1), b * c)

    ac = ps.AssignmentCollection([a1, a2], subexpressions=[])
    ac2 = ac.copy()
    assert ac2 == ac


def test_set_expressions():
    a1 = ps.Assignment(f[0, 0](0), a * b)
    a2 = ps.Assignment(f[0, 0](1), b * c)

    ac = ps.AssignmentCollection([a1, a2], subexpressions=[])

    ac.set_main_assignments_from_dict({d[0, 0](0): b * c})
    assert len(ac.main_assignments) == 1
    assert ac.main_assignments[0] == ps.Assignment(d[0, 0](0), b * c)

    ac.set_sub_expressions_from_dict({sp.symbols('xi'): a * b})
    assert len(ac.subexpressions) == 1
    assert ac.subexpressions[0] == ps.Assignment(sp.symbols('xi'), a * b)

    ac = ac.new_without_subexpressions(subexpressions_to_keep={sp.symbols('xi')})
    assert ac.subexpressions[0] == ps.Assignment(sp.symbols('xi'), a * b)

    ac = ac.new_without_unused_subexpressions()
    assert len(ac.subexpressions) == 0

    ac2 = ac.new_without_subexpressions()
    assert ac == ac2


def test_free_and_bound_symbols():
    a1 = ps.Assignment(a, d[0, 0](0))
    a2 = ps.Assignment(f[0, 0](1), b * c)

    ac = ps.AssignmentCollection([a2], subexpressions=[a1])
    assert f[0, 0](1) in ac.bound_symbols
    assert d[0, 0](0) in ac.free_symbols


def test_new_merged():
    a1 = ps.Assignment(a, b * c)
    a2 = ps.Assignment(a, x * y)
    a3 = ps.Assignment(t, x ** 2)

    # main assignments
    a4 = ps.Assignment(f[0, 0](0), a)
    a5 = ps.Assignment(d[0, 0](0), a)

    ac = ps.AssignmentCollection([a4], subexpressions=[a1])
    ac2 = ps.AssignmentCollection([a5], subexpressions=[a2, a3])

    merged_ac = ac.new_merged(ac2)

    assert len(merged_ac.subexpressions) == 3
    assert len(merged_ac.main_assignments) == 2
    assert ps.Assignment(sp.symbols('xi_0'), x * y) in merged_ac.subexpressions
    assert ps.Assignment(d[0, 0](0), sp.symbols('xi_0')) in merged_ac.main_assignments
    assert a1 in merged_ac.subexpressions
    assert a3 in merged_ac.subexpressions
