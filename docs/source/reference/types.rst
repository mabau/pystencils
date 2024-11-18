.. _page_type_system:

***********
Type System
***********

.. module:: pystencils.types


Type Creation and Conversion
----------------------------

.. autosummary::
    :toctree: autoapi
    :nosignatures:

    create_type
    create_numeric_type
    UserTypeSpec
    constify
    deconstify


Data Type Class Hierarchy
-------------------------

These are the classes that make up the type system internally.
Most of the time, you will not be using them directly, so you can skip over this part
unless you have very particular needs.


.. inheritance-diagram:: pystencils.types.meta.PsType pystencils.types.types
    :top-classes: pystencils.types.PsType
    :parts: 1

.. autosummary::
    :toctree: autoapi
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsType
    PsCustomType
    PsStructType
    PsDereferencableType
    PsPointerType
    PsArrayType
    PsNumericType
    PsScalarType
    PsVectorType
    PsIntegerType
    PsBoolType
    PsUnsignedIntegerType
    PsSignedIntegerType
    PsIeeeFloatType


Data Type Abbreviations
-----------------------

.. module:: pystencils.types.quick

The `pystencils.types.quick` module contains aliases of most of the above data type classes,
in order to reduce verbosity of code using the type system.

.. autosummary::
    
    Custom
    Scalar
    Ptr
    Arr
    Bool
    AnyInt
    UInt
    Int
    SInt
    Fp


Exceptions
----------

.. currentmodule:: pystencils.types

.. autosummary::
    :toctree: autoapi
    :nosignatures:

    pystencils.types.PsTypeError


Implementation Details
----------------------

.. automodule:: pystencils.types.meta
