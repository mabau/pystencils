# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pystencils
import pystencils.astnodes


def test_source_code_comment():

    a, b = pystencils.fields('a,b: float[2D]')

    assignments = pystencils.AssignmentCollection(
        {a.center(): b[0, 2] + b[0, 0]}, {}
    )

    ast = pystencils.create_kernel(assignments, target='cpu')

    ast.body.append(pystencils.astnodes.SourceCodeComment("Hallo"))
    ast.body.append(pystencils.astnodes.EmptyLine())
    ast.body.append(pystencils.astnodes.SourceCodeComment("World!"))
    print(ast)
    compiled = ast.compile()
    assert compiled is not None

    pystencils.show_code(ast)
