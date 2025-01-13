pystencils.sympyextensions
==========================

.. module:: pystencils.sympyextensions

Symbol Factory
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    SymbolCreator


Functions
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    math.prod
    math.scalar_product
    math.kronecker_delta
    math.tanh_step_function_approximation
    math.multidimensional_sum


Expression Analysis
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    math.is_constant
    math.summands
    math.common_denominator
    math.get_symmetric_part
    math.count_operations
    math.count_operations_in_ast


Expression Rewriting and Simplifications
----------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    math.remove_small_floats
    math.is_integer_sequence
    math.normalize_product
    math.symmetric_product
    math.fast_subs
    math.subs_additive
    math.replace_second_order_products
    math.remove_higher_order_terms
    math.complete_the_square
    math.complete_the_squares_in_exp
    math.extract_most_common_factor
    math.recursive_collect
    math.simplify_by_equality

Typed Expressions
-----------------

.. autoclass:: pystencils.TypedSymbol

.. autoclass:: pystencils.DynamicType
    :members:

.. autoclass:: pystencils.sympyextensions.CastFunc


Integer Operations
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/sympy_class.rst

    integer_functions.bitwise_xor
    integer_functions.bit_shift_right
    integer_functions.bit_shift_left
    integer_functions.bitwise_and
    integer_functions.bitwise_or
    integer_functions.int_div
    integer_functions.int_rem
    integer_functions.round_to_multiple_towards_zero
    integer_functions.ceil_to_multiple
    integer_functions.div_ceil
