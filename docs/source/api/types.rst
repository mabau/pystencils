***********
Type System
***********

.. automodule:: pystencils.types

Basic Functions
-------------------------------------

.. autofunction:: pystencils.types.create_type
.. autofunction:: pystencils.types.create_numeric_type
.. autofunction:: pystencils.types.constify
.. autofunction:: pystencils.types.deconstify



Data Type Class Hierarchy
-------------------------

.. inheritance-diagram:: pystencils.types.meta.PsType pystencils.types.types
    :top-classes: pystencils.types.PsType
    :parts: 1

.. autoclass:: pystencils.types.PsType
    :members:

.. automodule:: pystencils.types.types
    :members:


Data Type Abbreviations
-----------------------

.. automodule:: pystencils.types.quick
    :members:


Metaclass, Base Class and Uniquing Mechanisms
---------------------------------------------

.. automodule:: pystencils.types.meta

.. autoclass:: pystencils.types.meta.PsTypeMeta
    :members:

.. autofunction:: pystencils.types.PsType.__args__

.. autofunction:: pystencils.types.PsType.__canonical_args__
