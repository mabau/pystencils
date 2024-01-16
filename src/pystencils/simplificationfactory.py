from pystencils.simp import (SimplificationStrategy, insert_constants, insert_symbol_times_minus_one,
                             insert_constant_multiples, insert_constant_additions, insert_squares, insert_zeros)


def create_simplification_strategy():
    """
    Creates a default simplification `ps.simp.SimplificationStrategy`. The idea behind the default simplification
    strategy is to reduce the number of subexpressions by inserting single constants and to evaluate constant
    terms beforehand.
    """
    s = SimplificationStrategy()
    s.add(insert_symbol_times_minus_one)
    s.add(insert_constant_multiples)
    s.add(insert_constant_additions)
    s.add(insert_squares)
    s.add(insert_zeros)
    s.add(insert_constants)
    s.add(lambda ac: ac.new_without_unused_subexpressions())
