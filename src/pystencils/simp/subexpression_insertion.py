import sympy as sp
from pystencils.sympyextensions import is_constant

#   Subexpression Insertion


def insert_subexpressions(ac, selection_callback, skip=None):
    """
        Removes a number of subexpressions from an assignment collection by
        inserting their right-hand side wherever they occur.

        Args:
         - selection_callback: Function that is called to qualify subexpressions
            for insertion. Should return `True` for any subexpression that is to be
            inserted, and `False` otherwise.
         - skip: Set of symbols (left-hand sides of subexpressions) that should be 
            ignored even if qualified by the callback.
    """
    if skip is None:
        skip = set()
    i = 0
    while i < len(ac.subexpressions):
        exp = ac.subexpressions[i]
        if exp.lhs not in skip and selection_callback(exp):
            ac = ac.new_with_inserted_subexpression(exp.lhs)
        else:
            i += 1

    return ac


def insert_aliases(ac, **kwargs):
    """Inserts subexpressions that are aliases of other symbols, 
    i.e. their right-hand side is only another symbol."""
    return insert_subexpressions(ac, lambda x: isinstance(x.rhs, sp.Symbol), **kwargs)


def insert_zeros(ac, **kwargs):
    """Inserts subexpressions whose right-hand side is zero."""
    zero = sp.Integer(0)
    return insert_subexpressions(ac, lambda x: x.rhs == zero, **kwargs)


def insert_constants(ac, **kwargs):
    """Inserts subexpressions whose right-hand side is constant, 
    i.e. contains no symbols."""
    return insert_subexpressions(ac, lambda x: is_constant(x.rhs), **kwargs)


def insert_symbol_times_minus_one(ac, **kwargs):
    """Inserts subexpressions whose right-hand side is just a 
    negation of another symbol."""
    def callback(exp):
        rhs = exp.rhs
        minus_one = sp.Integer(-1)
        atoms = rhs.atoms(sp.Symbol)
        return len(atoms) == 1 and rhs == minus_one * atoms.pop()
    return insert_subexpressions(ac, callback, **kwargs)


def insert_constant_multiples(ac, **kwargs):
    """Inserts subexpressions whose right-hand side is a constant 
    multiplied with another symbol."""
    def callback(exp):
        rhs = exp.rhs
        symbols = rhs.atoms(sp.Symbol)
        numbers = rhs.atoms(sp.Number)
        return len(symbols) == 1 and len(numbers) == 1 and \
            rhs == numbers.pop() * symbols.pop()
    return insert_subexpressions(ac, callback, **kwargs)


def insert_constant_additions(ac, **kwargs):
    """Inserts subexpressions whose right-hand side is a sum of a 
    constant and another symbol."""
    def callback(exp):
        rhs = exp.rhs
        symbols = rhs.atoms(sp.Symbol)
        numbers = rhs.atoms(sp.Number)
        return len(symbols) == 1 and len(numbers) == 1 and \
            rhs == numbers.pop() + symbols.pop()
    return insert_subexpressions(ac, callback, **kwargs)


def insert_squares(ac, **kwargs):
    """Inserts subexpressions whose right-hand side is another symbol squared."""
    def callback(exp):
        rhs = exp.rhs
        symbols = rhs.atoms(sp.Symbol)
        return len(symbols) == 1 and rhs == symbols.pop() ** 2
    return insert_subexpressions(ac, callback, **kwargs)


def bind_symbols_to_skip(insertion_function, skip):
    return lambda ac: insertion_function(ac, skip=skip)
