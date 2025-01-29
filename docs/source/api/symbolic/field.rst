Fields
======

.. module:: pystencils.field

.. autoclass:: pystencils.Field

Types of Fields
---------------

.. autoclass:: pystencils.FieldType
    
Creating Fields
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    fields
    Field.create_generic
    Field.create_fixed_size
    Field.create_from_numpy_array
    Field.new_field_with_different_name
    
Properties
----------

Name and Element Type
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated

    Field.name
    Field.dtype
    Field.itemsize

Dimensionality, Shape, and Memory Layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    
    Field.ndim
    Field.values_per_cell
    Field.spatial_dimensions
    Field.index_dimensions
    Field.spatial_shape
    Field.has_fixed_shape
    Field.index_shape
    Field.has_fixed_index_shape
    Field.layout
    Field.spatial_strides
    Field.index_strides

Accessing Field Entries
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Field.center
    Field.center_vector
    Field.neighbor
    Field.neighbor_vector
    Field.__getitem__
    Field.__call__
    Field.absolute_access
    Field.staggered_access
    Field.staggered_vector_access

.. autoclass:: pystencils.field.Field.Access
