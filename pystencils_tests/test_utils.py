import sympy as sp
from pystencils.utils import LinearEquationSystem


def test_LinearEquationSystem():
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
