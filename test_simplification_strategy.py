import sympy as sp
from pystencils import Assignment, AssignmentCollection
from pystencils.assignment_collection import SimplificationStrategy, apply_on_all_subexpressions, \
    subexpression_substitution_in_existing_subexpressions


def test_simplification_strategy():
    a, b, c, d, x, y, z = sp.symbols("a b c d x y z")
    s0, s1, s2, s3 = sp.symbols("s_:4")
    a0, a1, a2, a3 = sp.symbols("a_:4")

    subexpressions = [
        Assignment(s0, 2*a + 2*b),
        Assignment(s1, 2 * a + 2 * b + 2*c),
        Assignment(s2, 2 * a + 2 * b + 2*c + 2*d),
    ]
    main = [
        Assignment(a0, s0 + s1),
        Assignment(a1, s0 + s2),
        Assignment(a2, s1 + s2),
    ]
    ac = AssignmentCollection(main, subexpressions)

    strategy = SimplificationStrategy()
    strategy.add(subexpression_substitution_in_existing_subexpressions)
    strategy.add(apply_on_all_subexpressions(sp.factor))

    result = strategy(ac)
    assert result.operation_count['adds'] == 7
    assert result.operation_count['muls'] == 5
    assert result.operation_count['divs'] == 0

    # Trigger display routines, such that they are at least executed
    report = strategy.show_intermediate_results(ac, symbols=[s0])
    assert 's_0' in str(report)
    report = strategy.show_intermediate_results(ac)
    assert 's_{1}' in report._repr_html_()

    report = strategy.create_simplification_report(ac)
    assert 'Adds' in str(report)
    assert 'Adds' in report._repr_html_()

    assert 'factor' in str(strategy)
