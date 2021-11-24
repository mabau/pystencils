# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import sympy as sp

import pystencils
from pystencils.typing import create_type


def test_floor_ceil_int_optimization():
    x, y = pystencils.fields('x,y: int32[2d]')
    a, b, c = sp.symbols('a, b, c')
    int_symbol = sp.Symbol('int_symbol', integer=True)
    typed_symbol = pystencils.TypedSymbol('typed_symbol', create_type('int64'))

    assignments = pystencils.AssignmentCollection({
        a:  sp.floor(1),
        b:  sp.ceiling(typed_symbol),
        c:  sp.floor(int_symbol),
        y.center():  sp.ceiling(x.center()) + sp.floor(x.center())
    })

    assert(typed_symbol.is_integer)
    print(sp.simplify(sp.ceiling(typed_symbol)))

    print(assignments)

    wild_floor = sp.floor(sp.Wild('w1'))

    assert not sp.floor(int_symbol).match(wild_floor)
    assert sp.floor(a).match(wild_floor)

    assert not assignments.find(wild_floor)


def test_floor_ceil_float_no_optimization():
    x, y = pystencils.fields('x,y: float32[2d]')
    a, b, c = sp.symbols('a, b, c')
    int_symbol = sp.Symbol('int_symbol', integer=True)
    typed_symbol = pystencils.TypedSymbol('typed_symbol', create_type('float32'))

    assignments = pystencils.AssignmentCollection({
        a:  sp.floor(1),
        b:  sp.ceiling(typed_symbol),
        c:  sp.floor(int_symbol),
        y.center():  sp.ceiling(x.center()) + sp.floor(x.center())
    })

    assert not typed_symbol.is_integer
    print(sp.simplify(sp.ceiling(typed_symbol)))

    print(assignments)

    wild_floor = sp.floor(sp.Wild('w1'))

    assert not sp.floor(int_symbol).match(wild_floor)
    assert sp.floor(a).match(wild_floor)

    assert assignments.find(wild_floor)
