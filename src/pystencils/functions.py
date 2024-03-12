import sympy as sp
from .types import PsPointerType


class AddressOf(sp.Function):
    """
    AddressOf is the '&' operation in C. It gets the address of a lvalue.
    """
    is_Atom = True

    def __new__(cls, arg):
        obj = sp.Function.__new__(cls, arg)
        return obj

    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()

    @property
    def is_commutative(self):
        return self.args[0].is_commutative

    @property
    def dtype(self):
        if hasattr(self.args[0], 'dtype'):
            return PsPointerType(self.args[0].dtype, const=True, restrict=True)
        else:
            raise ValueError(f'pystencils supports only non void pointers. Current address_of type: {self.args[0]}')
