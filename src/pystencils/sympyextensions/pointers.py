import sympy as sp
from ..types import PsPointerType, PsType


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
        arg_type = getattr(self.args[0], 'dtype', None)
        if arg_type is not None:
            assert isinstance(arg_type, PsType)
            return PsPointerType(arg_type, restrict=True, const=True)
        else:
            raise ValueError(f'pystencils supports only non void pointers. Current address_of type: {self.args[0]}')


class mem_acc(sp.Function):
    """Memory access through a raw pointer with an offset.
    
    This function should be used to model offset memory accesses through raw pointers.
    """
    
    @classmethod
    def eval(cls, ptr, offset):
        return None
