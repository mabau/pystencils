import pytest
import sympy as sp

from pystencils.utils import LinearEquationSystem


def test_linear_equation_system():
    unknowns = sp.symbols("x_:3")
    x, y, z = unknowns
    m = LinearEquationSystem(unknowns)
    m.add_equation(x + y - 2)
    m.add_equation(x - y - 1)
    assert m.solution_structure() == 'multiple'
    m.set_unknown_zero(2)
    assert m.solution_structure() == 'single'
    solution = m.solution()
    assert solution[unknowns[2]] == 0
    assert solution[unknowns[1]] == sp.Rational(1, 2)
    assert solution[unknowns[0]] == sp.Rational(3, 2)

    m.set_unknown_zero(0)
    assert m.solution_structure() == 'none'

    # special case where less rows than unknowns, but no solution
    m = LinearEquationSystem(unknowns)
    m.add_equation(x - 3)
    m.add_equation(x - 4)
    assert m.solution_structure() == 'none'
    m.add_equation(y - 4)
    assert m.solution_structure() == 'none'

    with pytest.raises(ValueError) as e:
        m.add_equation(x**2 - 1)
    assert 'Not a linear equation' in str(e)
