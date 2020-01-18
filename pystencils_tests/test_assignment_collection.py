import sympy as sp

from pystencils import Assignment, AssignmentCollection
from pystencils.astnodes import Conditional
from pystencils.simp.assignment_collection import SymbolGen


def test_assignment_collection():
    x, y, z, t = sp.symbols("x y z t")
    symbol_gen = SymbolGen("a")

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
    x, y, z, t = sp.symbols("x y z t")
    a, b = sp.symbols("a b")
    symbol_gen = SymbolGen("a")

    ac = AssignmentCollection([Assignment(z, x + y), Conditional(t > 0, Assignment(a, b+1), Assignment(a, b+2))],
                              [], subexpression_symbol_generator=symbol_gen)

    print(ac)
    print(ac.__repr__)


def test_vector_assignments():
    """From #17 (https://i10git.cs.fau.de/pycodegen/pystencils/issues/17)"""

    import pystencils as ps
    import sympy as sp
    a, b, c = sp.symbols("a b c")
    assignments = ps.Assignment(sp.Matrix([a,b,c]), sp.Matrix([1,2,3]))
    print(assignments)


def test_vector_assignment_collection():
    """From #17 (https://i10git.cs.fau.de/pycodegen/pystencils/issues/17)"""

    import pystencils as ps
    import sympy as sp
    a, b, c = sp.symbols("a b c")
    y, x = sp.Matrix([a,b,c]), sp.Matrix([1,2,3])
    assignments = ps.AssignmentCollection({y: x})
    print(assignments)

    assignments = ps.AssignmentCollection([ps.Assignment(y,x)])
    print(assignments)

