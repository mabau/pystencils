# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pytest

import pystencils
from pystencils.backends.cbackend import CBackend


class UnsupportedNode(pystencils.astnodes.Node):

    def __init__(self):
        super().__init__()


def test_print_unsupported_node():
    with pytest.raises(NotImplementedError, match='CBackend does not support node of type UnsupportedNode'):
        CBackend()(UnsupportedNode())
