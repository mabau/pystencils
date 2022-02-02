import sympy as sp

import pystencils as ps
from pystencils import Assignment, AssignmentCollection
from pystencils.simp import (
    SimplificationStrategy, apply_on_all_subexpressions,
    subexpression_substitution_in_existing_subexpressions)


def test_simplification_strategy():
    a, b, c, d, x, y, z = sp.symbols("a b c d x y z")
    s0, s1, s2, s3 = sp.symbols("s_:4")
    a0, a1, a2, a3 = sp.symbols("a_:4")

    subexpressions = [
        Assignment(s0, 2 * a + 2 * b),
        Assignment(s1, 2 * a + 2 * b + 2 * c),
        Assignment(s2, 2 * a + 2 * b + 2 * c + 2 * d),
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
    assert result.operation_count['muls'] == 4
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


def test_split_inner_loop():
    dst = ps.fields('dst(8): double[2D]')
    s = sp.symbols('s_:8')
    x = sp.symbols('x')
    subexpressions = []
    main = [
        Assignment(dst[0, 0](0), s[0]),
        Assignment(dst[0, 0](1), s[1]),
        Assignment(dst[0, 0](2), s[2]),
        Assignment(dst[0, 0](3), s[3]),
        Assignment(dst[0, 0](4), s[4]),
        Assignment(dst[0, 0](5), s[5]),
        Assignment(dst[0, 0](6), s[6]),
        Assignment(dst[0, 0](7), s[7]),
        Assignment(x, sum(s))
    ]
    ac = AssignmentCollection(main, subexpressions)
    split_groups = [[dst[0, 0](0), dst[0, 0](1)],
                    [dst[0, 0](2), dst[0, 0](3)],
                    [dst[0, 0](4), dst[0, 0](5)],
                    [dst[0, 0](6), dst[0, 0](7), x]]
    ac.simplification_hints['split_groups'] = split_groups
    ast = ps.create_kernel(ac)

    code = ps.get_code_str(ast)
    ps.show_code(ast)
    # we have four inner loops as indicated in split groups (4 elements) plus one outer loop
    assert code.count('for') == 5
    ast = ps.create_kernel(ac, target=ps.Target.GPU)

    code = ps.get_code_str(ast)
    # on GPUs is wouldn't be good to use loop splitting
    assert code.count('for') == 0

    ac = AssignmentCollection(main, subexpressions)
    ast = ps.create_kernel(ac)

    code = ps.get_code_str(ast)
    # one inner loop and one outer loop
    assert code.count('for') == 2
