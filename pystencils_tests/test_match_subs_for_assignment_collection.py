# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import sympy as sp

import pystencils
from pystencils.typing import TypedSymbol, BasicType


def test_wild_typed_symbol():
    x = pystencils.fields('x:  float32[3d]')
    typed_symbol = TypedSymbol('a', BasicType('float64'))

    assert x.center().match(sp.Wild('w1'))
    assert typed_symbol.match(sp.Wild('w1'))

    wild_ceiling = sp.ceiling(sp.Wild('w1'))
    assert sp.ceiling(x.center()).match(wild_ceiling)
    assert sp.ceiling(typed_symbol).match(wild_ceiling)


def test_replace_and_subs_for_assignment_collection():

    x, y = pystencils.fields('x, y:  float32[3d]')
    a, b, c, d = sp.symbols('a, b, c, d')

    assignments = pystencils.AssignmentCollection({
        a: sp.floor(1),
        b: 2,
        c: a + c,
        y.center(): sp.ceiling(x.center()) + sp.floor(x.center())
    })

    expected_assignments = pystencils.AssignmentCollection({
        a: sp.floor(3),
        b: 2,
        c: a + c,
        y.center(): sp.ceiling(x.center()) + sp.floor(x.center())
    })

    assert expected_assignments == assignments.replace(1, 3)
    assert expected_assignments == assignments.subs({1: 3})

    expected_assignments = pystencils.AssignmentCollection({
        d: sp.floor(1),
        b: 2,
        c: d + c,
        y.center(): sp.ceiling(x.center()) + sp.floor(x.center())
    })

    print(expected_assignments)
    print(assignments.subs(a, d))
    assert expected_assignments == assignments.subs(a, d)


def test_match_for_assignment_collection():

    x, y = pystencils.fields('x, y:  float32[3d]')
    a, b, c, d = sp.symbols('a, b, c, d')

    assignments = pystencils.AssignmentCollection({
        a: sp.floor(1),
        b: 2,
        c: a + c,
        y.center(): sp.ceiling(x.center()) + sp.floor(x.center())
    })

    w1 = sp.Wild('w1')
    w2 = sp.Wild('w2')
    w3 = sp.Wild('w3')

    wild_ceiling = sp.ceiling(w1)
    wild_addition = w1 + w2

    assert assignments.match(pystencils.Assignment(w3, wild_ceiling + w2))[w1] == x.center()
    assert assignments.match(pystencils.Assignment(w3, wild_ceiling + w2)) == {
        w3: y.center(),
        w2: sp.floor(x.center()),
        w1: x.center()
    }
    assert assignments.find(wild_ceiling) == {sp.ceiling(x.center())}
    assert len([a for a in assignments.find(wild_addition) if isinstance(a, sp.Add)]) == 2
