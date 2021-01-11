import pystencils as ps
from pystencils import TypedSymbol
from pystencils.astnodes import LoopOverCoordinate, SympyAssignment
from pystencils.data_types import create_type
from pystencils.transformations import filtered_tree_iteration, get_loop_hierarchy, get_loop_counter_symbol_hierarchy


def test_loop_information():
    f, g = ps.fields("f, g: double[2D]")
    update_rule = ps.Assignment(g[0, 0], f[0, 0])

    ast = ps.create_kernel(update_rule)
    inner_loops = [l for l in filtered_tree_iteration(ast, LoopOverCoordinate, stop_type=SympyAssignment)
                   if l.is_innermost_loop]

    loop_order = []
    for i in get_loop_hierarchy(inner_loops[0].args[0]):
        loop_order.append(i)

    assert loop_order == [0, 1]

    loop_symbols = get_loop_counter_symbol_hierarchy(inner_loops[0].args[0])

    assert loop_symbols == [TypedSymbol("ctr_1", create_type("int"), nonnegative=True),
                            TypedSymbol("ctr_0", create_type("int"), nonnegative=True)]
