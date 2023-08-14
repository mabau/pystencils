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
    assert len(block.args[1].body.args) == 1
    assert len(block.args[3].body.args) == 2

    for loop in loops:
        assert len(loop.parent.args) == 4  # 2 loops + 2 subexpressions
        assert loop.parent.args[0].lhs.name != loop.parent.args[2].lhs.name


def test_keep_order_of_accesses():
    f = ps.fields("f: [1D]")
    x = TypedSymbol("x", np.float64)
    n = 5

    loop = LoopOverCoordinate(Block([SympyAssignment(x, f[0]),
                                     SympyAssignment(f[1], 2 * x)]),
                              0, 0, n)
    block = Block([loop])

    ps.transformations.resolve_field_accesses(block)
    new_loops = ps.transformations.cut_loop(loop, [n - 1])
    ps.transformations.move_constants_before_loop(new_loops.args[1])

    kernel_func = ps.astnodes.KernelFunction(
        block, ps.Target.CPU, ps.Backend.C, ps.cpu.cpujit.make_python_function, None
    )
    kernel = kernel_func.compile()

    print(ps.show_code(kernel_func))

    f_arr = np.ones(n + 1)
    kernel(f=f_arr)

    print(f_arr)

    assert np.allclose(f_arr, np.array([
        1, 2, 4, 8, 16, 32
    ]))
