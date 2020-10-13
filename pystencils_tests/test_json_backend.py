# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import sympy

import pystencils
from pystencils.backends.json import print_json, print_yaml, write_json, write_yaml
import tempfile


def test_json_backend():

    z, y, x = pystencils.fields("z, y, x: [20,40]")
    a = sympy.Symbol('a')

    assignments = pystencils.AssignmentCollection({
        z[0, 0]: x[0, 0] * sympy.log(a * x[0, 0] * y[0, 0])
    })

    ast = pystencils.create_kernel(assignments)

    print(print_json(ast))
    print(print_yaml(ast))

    temp_dir = tempfile.TemporaryDirectory()
    write_json(temp_dir.name + '/test.json', ast)
    write_yaml(temp_dir.name + '/test.yaml', ast)
