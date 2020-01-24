"""

"""

import pytest

from pystencils.astnodes import Block
from pystencils.backends.cbackend import CustomCodeNode, get_headers


def test_headers_have_quotes_or_brackets():
    class ErrorNode1(CustomCodeNode):

        def __init__(self):
            super().__init__("", [], [])
            self.headers = ["iostream"]

    class ErrorNode2(CustomCodeNode):
        headers = ["<iostream>", "foo"]

        def __init__(self):
            super().__init__("", [], [])
            self.headers = ["<iostream>", "foo"]

    class OkNode3(CustomCodeNode):

        def __init__(self):
            super().__init__("", [], [])
            self.headers = ["<iostream>", '"foo"']

    with pytest.raises(AssertionError, match='.* does not follow the pattern .*'):
        get_headers(Block([ErrorNode1()]))

    with pytest.raises(AssertionError, match='.* does not follow the pattern .*'):
        get_headers(ErrorNode2())

    get_headers(OkNode3())
