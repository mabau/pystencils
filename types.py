import sympy as sp
from sympy.core.cache import cacheit


class TypedSymbol(sp.Symbol):

    def __new__(cls, name, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype):
        obj = super(TypedSymbol, cls).__xnew__(cls, name)
        obj._dtype = dtype
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self._dtype

    def _hashable_content(self):
        superClassContents = list(super(TypedSymbol, self)._hashable_content())
        t = tuple(superClassContents + [hash(self._dtype)])
        return t

    def __getnewargs__(self):
        return self.name, self.dtype


_c_dtype_dict = {0: 'int', 1: 'double', 2: 'float'}
_dtype_dict = {'int': 0, 'double': 1, 'float': 2}


class DataType(object):
    def __init__(self, dtype):
        self.alias = True
        self.const = False
        if isinstance(dtype, str):
            self.dtype = _dtype_dict[dtype]
        else:
            self.dtype = dtype

    def __repr__(self):
        return "{!s} {!s} {!s}".format("const" if self.const else "", "__restrict__" if not self.alias else "", _c_dtype_dict[self.dtype])
