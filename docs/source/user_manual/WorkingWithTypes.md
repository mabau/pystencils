---
file_format: mystnb
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

# Working with Data Types

This guide will demonstrate the various options that exist to customize the data types
in generated kernels.
Data types can be modified on different levels of granularity:
Individual fields and symbols,
single subexpressions,
or the entire kernel.

```{code-cell} ipython3
:tags: [remove-cell]
import pystencils as ps
import sympy as sp
```

## Changing the Default Data Types

The pystencils code generator defines two default data types:
 - The default *numeric type*, which is applied to all numerical computations that are not
   otherwise explicitly typed; the default is `float64`.
 - The default *index type*, which is used for all loop and field index calculations; the default is `int64`.

These can be modified by setting the
{any}`default_dtype <CreateKernelConfig.default_dtype>` and
{any}`index_type <CreateKernelConfig.index_dtype>`
options of the code generator configuration:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig()
cfg.default_dtype = "float32"
cfg.index_dtype = "int32"
```

Modifying these will change the way types for [untyped symbols](#untyped-symbols)
and [dynamically typed expressions](#dynamic-typing) are computed.

## Setting the Types of Fields and Symbols

(untyped-symbols)=
### Untyped Symbols

Symbols used inside a kernel are most commonly created using
{any}`sp.symbols <sympy.core.symbol.symbols>` or
{any}`sp.Symbol <sympy.core.symbol.Symbol>`.
These symbols are *untyped*; they will receive a type during code generation
according to these rules:
 - Free untyped symbols (i.e. symbols not defined by an assignment inside the kernel) receive the 
   {any}`default data type <CreateKernelConfig.default_dtype>` specified in the code generator configuration.
 - Bound untyped symbols (i.e. symbols that *are* defined in an assignment)
   receive the data type that was computed for the right-hand side expression of their defining assignment.

If you are working on kernels with homogenous data types, using untyped symbols will mostly be enough.

### Explicitly Typed Symbols and Fields

If you need more control over the data types in (parts of) your kernel,
you will have to explicitly specify them.
To set an explicit data type for a symbol, use the {any}`TypedSymbol` class of pystencils:

```{code-cell} ipython3
x_typed = ps.TypedSymbol("x", "uint32")
x_typed, str(x_typed.dtype)
```

You can set a `TypedSymbol` to any data type provided by [the type system](#page_type_system),
which will then be enforced by the code generator.

The same holds for fields:
When creating fields through the {any}`fields <pystencils.field.fields>` function,
add the type to the descriptor string; for instance:

```{code-cell} ipython3
f, g = ps.fields("f(1), g(3): float32[3D]")
str(f.dtype), str(g.dtype)
```

When using `Field.create_generic` or `Field.create_fixed_size`, on the other hand,
you can set the data type via the `dtype` keyword argument.

(dynamic-typing)=
### Dynamically Typed Symbols and Fields

Apart from explicitly setting data types,
`TypedSymbol`s and fields can also receive a *dynamic data type* (see {any}`DynamicType`).
There are two options:
 - Symbols or fields annotated with {any}`DynamicType.NUMERIC_TYPE` will always receive
   the {any}`default numeric type <CreateKernelConfig.default_dtype>` configured for the
   code generator.
   This is the default setting for fields
   created through `fields`, `Field.create_generic` or `Field.create_fixed_size`.
 - When annotated with {any}`DynamicType.INDEX_TYPE`, on the other hand, they will receive
   the {any}`index data type <CreateKernelConfig.index_dtype>` configured for the kernel.

Using dynamic typing, you can enforce symbols to receive either the standard numeric or
index type without explicitly stating it, such that your kernel definition becomes
independent from the code generator configuration.

## Mixing Types Inside Expressions

Pystencils enforces that all symbols, constants, and fields occuring inside an expression
have the same data type.
The code generator will never introduce implicit casts--if any type conflicts arise, it will terminate with an error.

Still, there are cases where you want to combine subexpressions of different types;
maybe you need to compute geometric information from loop counters or other integers,
or you are doing mixed-precision numerical computations.
In these cases, you might have to introduce explicit type casts when values move from one type context to another.
 
 <!-- 2. Annotate expressions with a specific data type to ensure computations are performed in that type. 
  TODO: See #97 (https://i10git.cs.fau.de/pycodegen/pystencils/-/issues/97)
 -->

(type_casts)=
### Type Casts

Type casts can be introduced into kernels using the {any}`tcast` symbolic function.
It takes an expression and a data type, which is either an explicit type (see [the type system](#page_type_system))
or a dynamic type ({any}`DynamicType`):

```{code-cell} ipython3
x, y = sp.symbols("x, y")
expr1 = ps.tcast(x, "float32")
expr2 = ps.tcast(3 + y, ps.DynamicType.INDEX_TYPE)

str(expr1.dtype), str(expr2.dtype)
```

When a type cast occurs, pystencils will compute the type of its argument independently
and then introduce a runtime cast to the target type.
That target type must comply with the type computed for the outer expression,
which the cast is embedded in.

## Understanding the pystencils Type Inference System

To correctly apply varying data types to pystencils kernels, it is important to understand
how pystencils computes and propagates the data types of symbols and expressions.

Type inference happens on the level of assignments.
For each assignment $x := \mathrm{calc}(y_1, \dots, y_n)$,
the system first attempts to compute a *unique* type for the right-hand side (RHS) $\mathrm{calc}(y_1, \dots, y_n)$.
It searches for any subexpression inside the RHS for which a type is already known --
these might be typed symbols
(whose types are either set explicitly by the user,
or have been determined from their defining assignment),
field accesses,
or explicitly typed expressions.
It then attempts to apply that data type to the entire expression.
If type conflicts occur, the process fails and the code generator raises an error.
Otherwise, the resulting type is assigned to the left-hand side symbol $x$.

:::{admonition} Developer's To Do
It would be great to illustrate this using a GraphViz-plot of an AST,
with nodes colored according to their data types
:::
