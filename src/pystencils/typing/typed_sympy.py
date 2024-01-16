from typing import Union

import numpy as np
import sympy as sp
from sympy.core.cache import cacheit

from pystencils.typing.types import BasicType, create_type, PointerType


def assumptions_from_dtype(dtype: Union[BasicType, np.dtype]):
    """Derives SymPy assumptions from :class:`BasicType` or a Numpy dtype

    Args:
        dtype (BasicType, np.dtype): a Numpy data type
    Returns:
        A dict of SymPy assumptions
    """
    if hasattr(dtype, 'numpy_dtype'):
        dtype = dtype.numpy_dtype

    assumptions = dict()

    try:
        if np.issubdtype(dtype, np.integer):
            assumptions.update({'integer': True})

        if np.issubdtype(dtype, np.unsignedinteger):
            assumptions.update({'negative': False})

        if np.issubdtype(dtype, np.integer) or \
                np.issubdtype(dtype, np.floating):
            assumptions.update({'real': True})
    except Exception:  # TODO this is dirty
        pass

    return assumptions


class TypedSymbol(sp.Symbol):
    def __new__(cls, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype, **kwargs):  # TODO does not match signature of sp.Symbol???
        # TODO: also Symbol should be allowed  ---> see sympy Variable
        assumptions = assumptions_from_dtype(dtype)
        assumptions.update(kwargs)
        obj = super(TypedSymbol, cls).__xnew__(cls, name, **assumptions)
        try:
            obj.numpy_dtype = create_type(dtype)
        except (TypeError, ValueError):
            # on error keep the string
            obj.numpy_dtype = dtype
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self.numpy_dtype

    def _hashable_content(self):
        return super()._hashable_content(), hash(self.numpy_dtype)

    def __getnewargs__(self):
        return self.name, self.dtype

    def __getnewargs_ex__(self):
        return (self.name, self.dtype), self.assumptions0

    @property
    def canonical(self):
        return self

    @property
    def reversed(self):
        return self

    @property
    def headers(self):
        headers = []
        try:
            if np.issubdtype(self.dtype.numpy_dtype, np.complexfloating):
                headers.append('"cuda_complex.hpp"')
        except Exception:
            pass
        try:
            if np.issubdtype(self.dtype.base_type.numpy_dtype, np.complexfloating):
                headers.append('"cuda_complex.hpp"')
        except Exception:
            pass

        return headers


SHAPE_DTYPE = BasicType('int64', const=True)
STRIDE_DTYPE = BasicType('int64', const=True)


class FieldStrideSymbol(TypedSymbol):
    """Sympy symbol representing the stride value of a field in a specific coordinate."""
    def __new__(cls, *args, **kwds):
        obj = FieldStrideSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_name, coordinate):
        name = f"_stride_{field_name}_{coordinate}"
        obj = super(FieldStrideSymbol, cls).__xnew__(cls, name, STRIDE_DTYPE, positive=True)
        obj.field_name = field_name
        obj.coordinate = coordinate
        return obj

    def __getnewargs__(self):
        return self.field_name, self.coordinate

    def __getnewargs_ex__(self):
        return (self.field_name, self.coordinate), {}

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def _hashable_content(self):
        return super()._hashable_content(), self.coordinate, self.field_name


class FieldShapeSymbol(TypedSymbol):
    """Sympy symbol representing the shape value of a sequence of fields. In a kernel iterating over multiple fields
    there is only one set of `FieldShapeSymbol`s since all the fields have to be of equal size."""
    def __new__(cls, *args, **kwds):
        obj = FieldShapeSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_names, coordinate):
        names = "_".join([field_name for field_name in field_names])
        name = f"_size_{names}_{coordinate}"
        obj = super(FieldShapeSymbol, cls).__xnew__(cls, name, SHAPE_DTYPE, positive=True)
        obj.field_names = tuple(field_names)
        obj.coordinate = coordinate
        return obj

    def __getnewargs__(self):
        return self.field_names, self.coordinate

    def __getnewargs_ex__(self):
        return (self.field_names, self.coordinate), {}

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def _hashable_content(self):
        return super()._hashable_content(), self.coordinate, self.field_names


class FieldPointerSymbol(TypedSymbol):
    """Sympy symbol representing the pointer to the beginning of the field data."""
    def __new__(cls, *args, **kwds):
        obj = FieldPointerSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_name, field_dtype, const):
        from pystencils.typing.utilities import get_base_type

        name = f"_data_{field_name}"
        dtype = PointerType(get_base_type(field_dtype), const=const, restrict=True)
        obj = super(FieldPointerSymbol, cls).__xnew__(cls, name, dtype)
        obj.field_name = field_name
        return obj

    def __getnewargs__(self):
        return self.field_name, self.dtype, self.dtype.const

    def __getnewargs_ex__(self):
        return (self.field_name, self.dtype, self.dtype.const), {}

    def _hashable_content(self):
        return super()._hashable_content(), self.field_name

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))
