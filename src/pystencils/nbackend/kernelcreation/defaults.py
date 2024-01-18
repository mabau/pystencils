"""This module defines various default types, symbols and variables for use in pystencils kernels.

On many occasions the SymPy frontend uses canonical symbols and types.
With the pymbolic-based backend, these symbols have to exist in two
variants; as `sp.Symbol` or `TypedSymbol`, and as `PsTypedVariable`s.
Therefore, for conciseness, this module should collect and provide each of these symbols.

We might furthermore consider making the defaults collection configurable.

A possibly incomplete list of symbols and types that need to be defined:

 - The default indexing data type (currently loosely defined as `int`)
 - The default spatial iteration counters (currently defined by `LoopOverCoordinate`)
 - The names of the coordinate members of index lists (currently in `CreateKernelConfig.coordinate_names`)
 - The sparse iteration counter (doesn't even exist yet)
 - ...
"""
