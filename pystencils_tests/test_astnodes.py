import pytest
import sys

import pystencils.config
import sympy as sp

import pystencils as ps
from pystencils import Assignment
from pystencils.astnodes import Block, LoopOverCoordinate, SkipIteration, SympyAssignment

dst = ps.fields('dst(8): double[2D]')
s = sp.symbols('s_:8')
x = sp.symbols('x')
y = sp.symbols('y')

python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def test_kernel_function():
    assignments = [
        Assignment(dst[0, 0](0), s[0]),
        Assignment(x, dst[0, 0](2))
    ]

    ast_node = ps.create_kernel(assignments)

    assert ast_node.target == ps.Target.CPU
    assert ast_node.backend == ps.Backend.C
    # symbols_defined and undefined_symbols will always return an emtpy set
    assert ast_node.symbols_defined == set()
    assert ast_node.undefined_symbols == set()
    assert ast_node.fields_written == {dst}
    assert ast_node.fields_read == {dst}


def test_skip_iteration():
    # skip iteration is an object which should give back empty data structures.
    skipped = SkipIteration()
    assert skipped.args == []
    assert skipped.symbols_defined == set()
    assert skipped.undefined_symbols == set()


def test_block():
    assignments = [
        Assignment(dst[0, 0](0), s[0]),
        Assignment(x, dst[0, 0](2))
    ]
    bl = Block(assignments)
    assert bl.symbols_defined == {dst[0, 0](0), dst[0, 0](2), s[0], x}

    bl.append([Assignment(y, 10)])
    assert bl.symbols_defined == {dst[0, 0](0), dst[0, 0](2), s[0], x, y}
    assert len(bl.args) == 3

    list_iterator = iter([Assignment(s[1], 11)])
    bl.insert_front(list_iterator)

    assert bl.args[0] == Assignment(s[1], 11)


def test_loop_over_coordinate():
    assignments = [
        Assignment(dst[0, 0](0), s[0]),
        Assignment(x, dst[0, 0](2))
    ]

    body = Block(assignments)
    loop = LoopOverCoordinate(body, coordinate_to_loop_over=0, start=0, stop=10, step=1)

    assert loop.body == body

    new_body = Block([assignments[0]])
    loop = loop.new_loop_with_different_body(new_body)
    assert loop.body == new_body

    assert loop.start == 0
    assert loop.stop == 10
    assert loop.step == 1

    loop.replace(loop.start, 2)
    loop.replace(loop.stop, 20)
    loop.replace(loop.step, 2)

    assert loop.start == 2
    assert loop.stop == 20
    assert loop.step == 2

