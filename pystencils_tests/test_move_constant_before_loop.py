import numpy as np

import pystencils as ps
from pystencils.astnodes import Block, LoopOverCoordinate, SympyAssignment, TypedSymbol
from pystencils.transformations import move_constants_before_loop


def test_symbol_renaming():
    """When two loops have assignments to the same symbol with different rhs and both
    are pulled before the loops, one of them has to be renamed
    """

    f, g = ps.fields("f, g : double[2D]")
    a, b, c = [TypedSymbol(n, np.float64) for n in ('a', 'b', 'c')]

    loop1 = LoopOverCoordinate(Block([SympyAssignment(c, a + b),
                                      SympyAssignment(g[0, 0], f[0, 0] + c)]),
                               0, 0, 10)
    loop2 = LoopOverCoordinate(Block([SympyAssignment(c, a ** 2 + b ** 2),
                                      SympyAssignment(g[0, 0], f[0, 0] + c)]),
                               0, 0, 10)
    block = Block([loop1, loop2])

    move_constants_before_loop(block)

    loops = block.atoms(LoopOverCoordinate)
    assert len(loops) == 2
    assert len(block.args[2].body.args) == 1
    assert len(block.args[3].body.args) == 2

    for loop in loops:
        assert len(loop.parent.args) == 4  # 2 loops + 2 subexpressions
        assert loop.parent.args[0].lhs.name != loop.parent.args[1].lhs.name
