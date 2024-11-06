import pytest
import sympy as sp
from pystencils.utils import LinearEquationSystem
from pystencils.utils import DotDict


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
    assert 'Not a linear equation' in str(e.value)

    x, y, z = sp.symbols("x, y, z")
    les = LinearEquationSystem([x, y, z])
    les.add_equation(1 * x + 2 * y - 1 * z + 4)
    les.add_equation(2 * x + 1 * y + 1 * z - 2)
    les.add_equation(1 * x + 2 * y + 1 * z + 2)

    # usually reduce is not necessary since almost every function of LinearEquationSystem calls reduce beforehand
    les.reduce()

    expected_matrix = sp.Matrix([[1, 0, 0, sp.Rational(5, 3)],
                                 [0, 1, 0, sp.Rational(-7, 3)],
                                 [0, 0, 1, sp.Integer(1)]])
    assert les.matrix == expected_matrix
    assert les.rank == 3

    sol = les.solution()
    assert sol[x] == sp.Rational(5, 3)
    assert sol[y] == sp.Rational(-7, 3)
    assert sol[z] == sp.Integer(1)

    les = LinearEquationSystem([x, y])
    assert les.solution_structure() == 'multiple'

    les.add_equation(x + 1)
    assert les.solution_structure() == 'multiple'

    les.add_equation(y + 2)
    assert les.solution_structure() == 'single'

    les.add_equation(x + y + 5)
    assert les.solution_structure() == 'none'


def test_dot_dict():
    d = {'a': {'c': 7}, 'b': 6}
    t = DotDict(d)
    assert t.a.c == 7
    assert t.b == 6
    assert len(t) == 2

    delattr(t, 'b')
    assert len(t) == 1

    t.b = 6
    assert len(t) == 2
    assert t.b == 6
