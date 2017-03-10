import sympy as sp
from sympy.core.cache import cacheit


class TypedSymbol(sp.Symbol):

    def __new__(cls, name, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype):
        obj = super(TypedSymbol, cls).__xnew__(cls, name)
        obj._dtype = DataType(dtype) if isinstance(dtype, str) else dtype
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self._dtype

    def _hashable_content(self):
        superClassContents = list(super(TypedSymbol, self)._hashable_content())
        t = tuple(superClassContents + [hash(repr(self._dtype))])
        return t

    def __getnewargs__(self):
        return self.name, self.dtype


_c_dtype_dict = {0: 'bool', 1: 'int', 2: 'float', 3: 'double'}
_dtype_dict = {'bool': 0, 'int': 1, 'float': 2, 'double': 3}


class DataType(object):
    def __init__(self, dtype):
        self.alias = True
        self.const = False
        self.ptr = False
        self.dtype = 0
        if isinstance(dtype, str):
            for s in dtype.split():
                if s == 'const':
                    self.const = True
                elif s == '*':
                    self.ptr = True
                elif s == '__restrict__':
                    self.alias = False
                else:
                    self.dtype = _dtype_dict[s]
        elif isinstance(dtype, DataType):
            self.__dict__.update(dtype.__dict__)
        else:
            self.dtype = dtype

    def __repr__(self):
        return "{!s} {!s}{!s} {!s}".format("const" if self.const else "", _c_dtype_dict[self.dtype],
                                           "*" if self.ptr else "", "__restrict__" if not self.alias else "")

    def __eq__(self, other):
        if self.alias == other.alias and self.const == other.const and self.ptr == other.ptr and self.dtype == other.dtype:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.ptr and not other.ptr:
            return True
        if self.dtype > other.dtype:
            return True

    def __hash__(self):
        return hash(repr(self))


def get_type_from_sympy(node):
    # Rational, NumberSymbol?
    # Zero, One, NegativeOne )= Integer
    # Half )= Rational
    # NAN, Infinity, Negative Inifinity,
    # Exp1, Imaginary Unit, Pi, EulerGamma, Catalan, Golden Ratio
    # Pow, Mul, Add, Mod, Relational
    if not isinstance(node, sp.Number):
        raise TypeError(node, 'is not a sp.Number')

    if isinstance(node, sp.Float) or isinstance(node, sp.RealNumber):
        return DataType('double'), float(node)
    elif isinstance(node, sp.Integer):
        return DataType('int'), int(node)
    elif isinstance(node, sp.Rational):
        raise NotImplementedError('Rationals are not supported yet')
    else:
        raise TypeError(node, ' is not a supported type!')
