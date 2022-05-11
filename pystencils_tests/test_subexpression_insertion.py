from pystencils import fields, Assignment, AssignmentCollection
from pystencils.simp.subexpression_insertion import *


def test_subexpression_insertion():
    f, g = fields('f(10), g(10) : [2D]')
    xi = sp.symbols('xi_:10')
    xi_set = set(xi)

    subexpressions = [
        Assignment(xi[0], -f(4)),
        Assignment(xi[1], -(f(1) * f(2))),
        Assignment(xi[2], 2.31 * f(5)),
        Assignment(xi[3], 1.8 + f(5) + f(6)),
        Assignment(xi[4], 5.7 + f(6)),
        Assignment(xi[5], (f(4) + f(5))**2),
        Assignment(xi[6], f(3)**2),
        Assignment(xi[7], f(4)),
        Assignment(xi[8], 13),
        Assignment(xi[9], 0),
    ]

    assignments = [Assignment(g(i), x) for i, x in enumerate(xi)]
    ac = AssignmentCollection(assignments, subexpressions=subexpressions)

    ac_ins = insert_symbol_times_minus_one(ac)
    assert (ac_ins.bound_symbols & xi_set) == (xi_set - {xi[0]})

    ac_ins = insert_constant_multiples(ac)
    assert (ac_ins.bound_symbols & xi_set) == (xi_set - {xi[0], xi[2]})

    ac_ins = insert_constant_additions(ac)
    assert (ac_ins.bound_symbols & xi_set) == (xi_set - {xi[4]})

    ac_ins = insert_squares(ac)
    assert (ac_ins.bound_symbols & xi_set) == (xi_set - {xi[6]})

    ac_ins = insert_aliases(ac)
    assert (ac_ins.bound_symbols & xi_set) == (xi_set - {xi[7]})

    ac_ins = insert_zeros(ac)
    assert (ac_ins.bound_symbols & xi_set) == (xi_set - {xi[9]})

    ac_ins = insert_constants(ac, skip={xi[9]})
    assert (ac_ins.bound_symbols & xi_set) == (xi_set - {xi[8]})
