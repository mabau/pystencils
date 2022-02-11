import sympy as sp
from pystencils.typing import PointerType


class DivFunc(sp.Function):
    """
    DivFunc represents a division operation, since sympy represents divisions with ^-1
    """
    is_Atom = True
    is_real = True

    def __new__(cls, *args, **kwargs):
        if len(args) != 2:
            raise ValueError(f'{cls} takes only 2 arguments, instead {len(args)} received!')
        divisor, dividend, *other_args = args

        return sp.Function.__new__(cls, divisor, dividend, *other_args, **kwargs)

    def _eval_evalf(self, *args, **kwargs):
        return self.divisor.evalf() / self.dividend.evalf()

    @property
    def divisor(self):
        return self.args[0]

    @property
    def dividend(self):
        return self.args[1]


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
            return PointerType(self.args[0].dtype, restrict=True)
        else:
            raise ValueError(f'pystencils supports only non void pointers. Current address_of type: {self.args[0]}')
