****************************************
Constants, Memory Objects, and Functions
****************************************

Memory Objects: Symbols and Buffers
===================================

The Memory Model
----------------

In order to reason about memory accesses, mutability, invariance, and aliasing, the *pystencils* backend uses
a very simple memory model. There are three types of memory objects:

- Symbols (`PsSymbol`), which act as registers for data storage within the scope of a kernel
- Field buffers (`PsBuffer`), which represent a contiguous block of memory the kernel has access to, and
- the *unmanaged heap*, which is a global catch-all memory object which all pointers not belonging to a field
  array point into.

All of these objects are disjoint, and cannot alias each other.
Each symbol exists in isolation,
field buffers do not overlap,
and raw pointers are assumed not to point into memory owned by a symbol or field array.
Instead, all raw pointers point into unmanaged heap memory, and are assumed to *always* alias one another:
Each change brought to unmanaged memory by one raw pointer is assumed to affect the memory pointed to by
another raw pointer.

Symbols
-------

In the pystencils IR, instances of `PsSymbol` represent what is generally known as "virtual registers".
These are memory locations that are private to a function, cannot be aliased or pointed to, and will finally reside
either in physical registers or on the stack.
Each symbol has a name and a data type. The data type may initially be `None`, in which case it should soon after be
determined by the `Typifier`.

Other than their front-end counterpart `sympy.Symbol <sympy.core.symbol.Symbol>`, `PsSymbol` instances are mutable;
their properties can and often will change over time.
As a consequence, they are not comparable by value:
two `PsSymbol` instances with the same name and data type will in general *not* be equal.
In fact, most of the time, it is an error to have two identical symbol instances active.

Creating Symbols
^^^^^^^^^^^^^^^^

During kernel translation, symbols never exist in isolation, but should always be managed by a `KernelCreationContext`.
Symbols can be created and retrieved using `add_symbol <KernelCreationContext.add_symbol>` and `find_symbol <KernelCreationContext.find_symbol>`.
A symbol can also be duplicated using `duplicate_symbol <KernelCreationContext.duplicate_symbol>`, which assigns a new name to the symbol's copy.
The `KernelCreationContext` keeps track of all existing symbols during a kernel translation run
and makes sure that no name and data type conflicts may arise.

Never call the constructor of `PsSymbol` directly unless you really know what you are doing.

Symbol Properties
^^^^^^^^^^^^^^^^^

Symbols can be annotated with arbitrary information using *symbol properties*.
Each symbol property type must be a subclass of `PsSymbolProperty`.
It is strongly recommended to implement property types using frozen
`dataclasses <https://docs.python.org/3/library/dataclasses.html>`_.
For example, this snippet defines a property type that models pointer alignment requirements:

.. code-block:: python

    @dataclass(frozen=True)
    class AlignmentProperty(UniqueSymbolProperty)
        """Require this pointer symbol to be aligned at a particular byte boundary."""
        
        byte_boundary: int

Inheriting from `UniqueSymbolProperty` ensures that at most one property of this type can be attached to
a symbol at any time.
Properties can be added, queried, and removed using the `PsSymbol` properties API listed below.

Many symbol properties are more relevant to consumers of generated kernels than to the code generator itself.
The above alignment property, for instance, may be added to a pointer symbol by a vectorization pass
to document its assumption that the pointer be properly aligned, in order to emit aligned load and store instructions.
It then becomes the responsibility of the runtime system embedding the kernel to check this prequesite before calling the kernel.
To make sure this information becomes visible, any properties attached to symbols exposed as kernel parameters will also
be added to their respective `KernelParameter` instance.

Buffers
-------

Buffers, as represented by the `PsBuffer` class, represent contiguous, n-dimensional, linearized cuboid blocks of memory.
Each buffer has a fixed name and element data type,
and will be represented in the IR via three sets of symbols:

- The *base pointer* is a symbol of pointer type which points into the buffer's underlying memory area.
  Each buffer has at least one, its primary base pointer, whose pointed-to type must be the same as the
  buffer's element type. There may be additional base pointers pointing into subsections of that memory.
  These additional base pointers may also have deviating data types, as is for instance required for
  type erasure in certain cases.
  To communicate its role to the code generation system,
  each base pointer needs to be marked as such using the `BufferBasePtr` property,
  .
- The buffer *shape* defines the size of the buffer in each dimension. Each shape entry is either a `symbol <PsSymbol>`
  or a `constant <PsConstant>`.
- The buffer *strides* define the step size to go from one entry to the next in each dimension.
  Like the shape, each stride entry is also either a symbol or a constant.

The shape and stride symbols must all have the same data type, which will be stored as the buffer's index data type.

Creating and Managing Buffers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarily to symbols, buffers are typically managed by the `KernelCreationContext`, which associates each buffer
to a front-end `Field`. Buffers for fields can be obtained using `get_buffer <KernelCreationContext.get_buffer>`.
The context makes sure to avoid name conflicts between buffers.

API Documentation
=================

.. automodule:: pystencils.backend.properties
    :members:

.. automodule:: pystencils.backend.memory
    :members:

.. automodule:: pystencils.backend.constants
    :members:

.. autoclass:: pystencils.backend.literals.PsLiteral
    :members:

.. automodule:: pystencils.backend.functions
