class NestedScopes:
    """Symbol visibility model using nested scopes

    - every accessed symbol that was not defined before, is added as a "free parameter"
    - free parameters are global, i.e. they are not in scopes
    - push/pop adds or removes a scope

    >>> s = NestedScopes()
    >>> s.access_symbol("a")
    >>> s.is_defined("a")
    False
    >>> s.free_parameters
    {'a'}
    >>> s.define_symbol("b")
    >>> s.is_defined("b")
    True
    >>> s.push()
    >>> s.is_defined_locally("b")
    False
    >>> s.define_symbol("c")
    >>> s.pop()
    >>> s.is_defined("c")
    False
    """

    def __init__(self):
        self.free_parameters = set()
        self._defined = [set()]

    def access_symbol(self, symbol):
        if not self.is_defined(symbol):
            self.free_parameters.add(symbol)

    def define_symbol(self, symbol):
        self._defined[-1].add(symbol)

    def is_defined(self, symbol):
        return any(symbol in scopes for scopes in self._defined)

    def is_defined_locally(self, symbol):
        return symbol in self._defined[-1]

    def push(self):
        self._defined.append(set())

    def pop(self):
        self._defined.pop()
        assert self.depth >= 1

    @property
    def depth(self):
        return len(self._defined)
