"""Special symbols representing kernel parameters related to fields/arrays.

A `KernelFunction` node determines parameters that have to be passed to the function by searching for all undefined
symbols. Some symbols are not directly defined by the user, but are related to the `Field`s used in the kernel:
For each field a `FieldPointerSymbol` needs to be passed in, which is the pointer to the memory region where
the field is stored. This pointer is represented by the `FieldPointerSymbol` class that additionally stores the
name of the corresponding field. For fields where the size is not known at compile time, additionally shape and stride
information has to be passed in at runtime. These values are represented by  `FieldShapeSymbol`
and `FieldPointerSymbol`.

The special symbols in this module store only the field name instead of a field reference. Storing a field reference
directly leads to problems with copying and pickling behaviour due to the circular dependency of `Field` and
e.g. `FieldShapeSymbol`, since a Field contains `FieldShapeSymbol`s in its shape, and a `FieldShapeSymbol`
would reference back to the field.
"""
from sympy.core.cache import cacheit

from pystencils.data_types import (
    PointerType, TypedSymbol, create_composite_type_from_string, get_base_type)

SHAPE_DTYPE = create_composite_type_from_string("const int64")
STRIDE_DTYPE = create_composite_type_from_string("const int64")


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
