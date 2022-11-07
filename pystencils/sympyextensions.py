import itertools
import operator
import warnings
from collections import Counter, defaultdict
from functools import partial, reduce
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import sympy as sp
from sympy import PolynomialError
from sympy.functions import Abs
from sympy.core.numbers import Zero

from pystencils.assignment import Assignment
from pystencils.functions import DivFunc
from pystencils.typing import CastFunc, get_type_of_expression, PointerType, VectorType
from pystencils.typing.typed_sympy import FieldPointerSymbol

T = TypeVar('T')


def prod(seq: Iterable[T]) -> T:
    """Takes a sequence and returns the product of all elements"""
    return reduce(operator.mul, seq, 1)


def remove_small_floats(expr, threshold):
    """Removes all sp.Float objects whose absolute value is smaller than threshold

    >>> expr = sp.sympify("x + 1e-15 * y")
    >>> remove_small_floats(expr, 1e-14)
    x
    """
    if isinstance(expr, sp.Float) and sp.Abs(expr) < threshold:
        return 0
    else:
        new_args = [remove_small_floats(c, threshold) for c in expr.args]
        return expr.func(*new_args) if new_args else expr


def is_integer_sequence(sequence: Iterable) -> bool:
    """Checks if all elements of the passed sequence can be cast to integers"""
    try:
        for i in sequence:
            int(i)
        return True
    except TypeError:
        return False


def scalar_product(a: Iterable[T], b: Iterable[T]) -> T:
    """Scalar product between two sequences."""
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def kronecker_delta(*args):
    """Kronecker delta for variable number of arguments, 1 if all args are equal, otherwise 0"""
    for a in args:
        if a != args[0]:
            return 0
    return 1


def tanh_step_function_approximation(x, step_location, kind='right', steepness=0.0001):
    """Approximation of step function by a tanh function

    >>> tanh_step_function_approximation(1.2, step_location=1.0, kind='right')
    1.00000000000000
    >>> tanh_step_function_approximation(0.9, step_location=1.0, kind='right')
    0
    >>> tanh_step_function_approximation(1.1, step_location=1.0, kind='left')
    0
    >>> tanh_step_function_approximation(0.9, step_location=1.0, kind='left')
    1.00000000000000
    >>> tanh_step_function_approximation(0.5, step_location=(0, 1), kind='middle')
    1
    """
    if kind == 'left':
        return (1 - sp.tanh((x - step_location) / steepness)) / 2
    elif kind == 'right':
        return (1 + sp.tanh((x - step_location) / steepness)) / 2
    elif kind == 'middle':
        x1, x2 = step_location
        return 1 - (tanh_step_function_approximation(x, x1, 'left', steepness)
                    + tanh_step_function_approximation(x, x2, 'right', steepness))


def multidimensional_sum(i, dim):
    """Multidimensional summation

    Example:
        >>> list(multidimensional_sum(2, dim=3))
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    """
    prod_args = [range(dim)] * i
    return itertools.product(*prod_args)


def normalize_product(product: sp.Expr) -> List[sp.Expr]:
    """Expects a sympy expression that can be interpreted as a product and returns a list of all factors.

    Removes sp.Pow nodes that have integer exponent by representing them as single factors in list.

    Returns:
        * for a Mul node list of factors ('args')
        * for a Pow node with positive integer exponent a list of factors
        * for other node types [product] is returned
    """
    def handle_pow(power):
        if power.exp.is_integer and power.exp.is_number and power.exp > 0:
            return [power.base] * power.exp
        else:
            return [power]

    if isinstance(product, sp.Pow):
        return handle_pow(product)
    elif isinstance(product, sp.Mul):
        result = []
        for a in product.args:
            if a.func == sp.Pow:
                result += handle_pow(a)
            else:
                result.append(a)
        return result
    else:
        return [product]


def symmetric_product(*args, with_diagonal: bool = True) -> Iterable:
    """Similar to itertools.product but yields only values where the index is ascending i.e. values below/up to diagonal

    Examples:
        >>> list(symmetric_product([1, 2, 3], ['a', 'b', 'c']))
        [(1, 'a'), (1, 'b'), (1, 'c'), (2, 'b'), (2, 'c'), (3, 'c')]
        >>> list(symmetric_product([1, 2, 3], ['a', 'b', 'c'], with_diagonal=False))
        [(1, 'b'), (1, 'c'), (2, 'c')]
    """
    ranges = [range(len(a)) for a in args]
    for idx in itertools.product(*ranges):
        valid_index = True
        for t in range(1, len(idx)):
            if (with_diagonal and idx[t - 1] > idx[t]) or (not with_diagonal and idx[t - 1] >= idx[t]):
                valid_index = False
                break
        if valid_index:
            yield tuple(a[i] for a, i in zip(args, idx))


def fast_subs(expression: T, substitutions: Dict,
              skip: Optional[Callable[[sp.Expr], bool]] = None) -> T:
    """Similar to sympy subs function.

    Args:
        expression: expression where parts should be substituted
        substitutions: dict defining substitutions by mapping from old to new terms
        skip: function that marks expressions to be skipped (if True is returned) - that means that in these skipped
              expressions no substitutions are done

    This version is much faster for big substitution dictionaries than sympy version
    """
    if type(expression) is sp.Matrix:
        return expression.copy().applyfunc(partial(fast_subs, substitutions=substitutions))

    def visit(expr, evaluate=True):
        if skip and skip(expr):
            return expr
        elif hasattr(expr, "fast_subs"):
            return expr.fast_subs(substitutions, skip)
        elif expr in substitutions:
            return substitutions[expr]
        elif not hasattr(expr, 'args'):
            return expr
        elif isinstance(expr, (sp.UnevaluatedExpr, DivFunc)):
            args = [visit(a, False) for a in expr.args]
            return expr.func(*args)
        else:
            param_list = [visit(a, evaluate) for a in expr.args]
            if isinstance(expr, (sp.Mul, sp.Add)):
                return expr if not param_list else expr.func(*param_list, evaluate=evaluate)
            return expr if not param_list else expr.func(*param_list)

    if len(substitutions) == 0:
        return expression
    else:
        return visit(expression)


def is_constant(expr):
    """Simple version of checking if a sympy expression is constant.
    Works also for piecewise defined functions - sympy's is_constant() has a problem there, see:
    https://github.com/sympy/sympy/issues/16662
    """
    return len(expr.free_symbols) == 0


def subs_additive(expr: sp.Expr, replacement: sp.Expr, subexpression: sp.Expr,
                  required_match_replacement: Optional[Union[int, float]] = 0.5,
                  required_match_original: Optional[Union[int, float]] = None) -> sp.Expr:
    """Transformation for replacing a given subexpression inside a sum.

    Examples:
        The next example demonstrates the advantage of replace_additive compared to sympy.subs:
        >>> x, y, z, k = sp.symbols("x y z k")
        >>> subs_additive(3*x + 3*y, replacement=k, subexpression=x + y)
        3*k

        Terms that don't match completely can be substituted at the cost of additional terms.
        This trade-off is managed using the required_match parameters.
        >>> subs_additive(3*x + 3*y + z, replacement=k, subexpression=x+y+z, required_match_original=1.0)
        3*x + 3*y + z
        >>> subs_additive(3*x + 3*y + z, replacement=k, subexpression=x+y+z, required_match_original=0.5)
        3*k - 2*z
        >>> subs_additive(3*x + 3*y + z, replacement=k, subexpression=x+y+z, required_match_original=2)
        3*k - 2*z

    Args:
        expr: input expression
        replacement: expression that is inserted for subexpression (if found)
        subexpression: expression to replace
        required_match_replacement:
             * if float: the percentage of terms of the subexpression that has to be matched in order to replace
             * if integer: the total number of terms that has to be matched in order to replace
             * None: is equal to integer 1
             * if both match parameters are given, both restrictions have to be fulfilled (i.e. logical AND)
        required_match_original:
             * if float: the percentage of terms of the original addition expression that has to be matched
             * if integer: the total number of terms that has to be matched in order to replace
             * None: is equal to integer 1

    Returns:
        new expression with replacement
    """
    def normalize_match_parameter(match_parameter, expression_length):
        if match_parameter is None:
            return 1
        elif isinstance(match_parameter, float):
            assert 0 <= match_parameter <= 1
            res = int(match_parameter * expression_length)
            return max(res, 1)
        elif isinstance(match_parameter, int):
            assert match_parameter > 0
            return match_parameter
        raise ValueError("Invalid parameter")

    normalized_replacement_match = normalize_match_parameter(required_match_replacement, len(subexpression.args))

    if isinstance(subexpression, sp.Number):
        return expr.subs({replacement: subexpression})

    def visit(current_expr):
        if current_expr.is_Add:
            expr_max_length = max(len(current_expr.args), len(subexpression.args))
            normalized_current_expr_match = normalize_match_parameter(required_match_original, expr_max_length)
            expr_coefficients = current_expr.as_coefficients_dict()
            subexpression_coefficient_dict = subexpression.as_coefficients_dict()
            intersection = set(subexpression_coefficient_dict.keys()).intersection(set(expr_coefficients))
            if len(intersection) >= max(normalized_replacement_match, normalized_current_expr_match):
                # find common factor
                factors = defaultdict(int)
                skips = 0
                for common_symbol in subexpression_coefficient_dict.keys():
                    if common_symbol not in expr_coefficients:
                        skips += 1
                        continue
                    factor = expr_coefficients[common_symbol] / subexpression_coefficient_dict[common_symbol]
                    factors[sp.simplify(factor)] += 1

                common_factor = max(factors.items(), key=operator.itemgetter(1))[0]
                if factors[common_factor] >= max(normalized_current_expr_match, normalized_replacement_match):
                    return current_expr - common_factor * subexpression + common_factor * replacement

        # if no subexpression was found
        param_list = [visit(a) for a in current_expr.args]
        if not param_list:
            return current_expr
        else:
            if current_expr.func == sp.Mul and Zero() in param_list:
                return sp.simplify(current_expr)
            else:
                return current_expr.func(*param_list, evaluate=False)

    return visit(expr)


def replace_second_order_products(expr: sp.Expr, search_symbols: Iterable[sp.Symbol],
                                  positive: Optional[bool] = None,
                                  replace_mixed: Optional[List[Assignment]] = None) -> sp.Expr:
    """Replaces second order mixed terms like 4*x*y by 2*( (x+y)**2 - x**2 - y**2 ).

    This makes the term longer - simplify usually is undoing these - however this
    transformation can be done to find more common sub-expressions

    Args:
        expr: input expression
        search_symbols: symbols that are searched for
                         for example, given [x,y,z] terms like x*y, x*z, z*y are replaced
        positive: there are two ways to do this substitution, either with term
                 (x+y)**2 or (x-y)**2 . if positive=True the first version is done,
                 if positive=False the second version is done, if positive=None the
                 sign is determined by the sign of the mixed term that is replaced
        replace_mixed: if a list is passed here, the expr x+y or x-y is replaced by a special new symbol
                       and the replacement equation is added to the list
    """
    mixed_symbols_replaced = set([e.lhs for e in replace_mixed]) if replace_mixed is not None else set()

    if expr.is_Mul:
        distinct_search_symbols = set()
        nr_of_search_terms = 0
        other_factors = sp.Integer(1)
        for t in expr.args:
            if t in search_symbols:
                nr_of_search_terms += 1
                distinct_search_symbols.add(t)
            else:
                other_factors *= t
        if len(distinct_search_symbols) == 2 and nr_of_search_terms == 2:
            u, v = sorted(list(distinct_search_symbols), key=lambda symbol: symbol.name)
            if positive is None:
                other_factors_without_symbols = other_factors
                for s in other_factors.atoms(sp.Symbol):
                    other_factors_without_symbols = other_factors_without_symbols.subs(s, 1)
                positive = other_factors_without_symbols.is_positive
                assert positive is not None
            sign = 1 if positive else -1
            if replace_mixed is not None:
                new_symbol_str = 'P' if positive else 'M'
                mixed_symbol_name = u.name + new_symbol_str + v.name
                mixed_symbol = sp.Symbol(mixed_symbol_name.replace("_", ""))
                if mixed_symbol not in mixed_symbols_replaced:
                    mixed_symbols_replaced.add(mixed_symbol)
                    replace_mixed.append(Assignment(mixed_symbol, u + sign * v))
            else:
                mixed_symbol = u + sign * v
            return sp.Rational(1, 2) * sign * other_factors * (mixed_symbol ** 2 - u ** 2 - v ** 2)

    param_list = [replace_second_order_products(a, search_symbols, positive, replace_mixed) for a in expr.args]
    result = expr.func(*param_list, evaluate=False) if param_list else expr
    return result


def remove_higher_order_terms(expr: sp.Expr, symbols: Sequence[sp.Symbol], order: int = 3) -> sp.Expr:
    """Removes all terms that contain more than 'order' factors of given 'symbols'

    Example:
        >>> x, y = sp.symbols("x y")
        >>> term = x**2 * y + y**2 * x + y**3 + x + y ** 2
        >>> remove_higher_order_terms(term, order=2, symbols=[x, y])
        x + y**2
    """
    from sympy.core.power import Pow
    from sympy.core.add import Add, Mul

    result = 0
    expr = expr.expand()

    def velocity_factors_in_product(product):
        factor_count = 0
        if type(product) is Mul:
            for factor in product.args:
                if type(factor) == Pow:
                    if factor.args[0] in symbols:
                        factor_count += factor.args[1]
                if factor in symbols:
                    factor_count += 1
        elif type(product) is Pow:
            if product.args[0] in symbols:
                factor_count += product.args[1]
        return factor_count

    if type(expr) == Mul or type(expr) == Pow:
        if velocity_factors_in_product(expr) <= order:
            return expr
        else:
            return Zero()

    if type(expr) != Add:
        return expr

    for sum_term in expr.args:
        if velocity_factors_in_product(sum_term) <= order:
            result += sum_term
    return result


def complete_the_square(expr: sp.Expr, symbol_to_complete: sp.Symbol,
                        new_variable: sp.Symbol) -> Tuple[sp.Expr, Optional[Tuple[sp.Symbol, sp.Expr]]]:
    """Transforms second order polynomial into only squared part.

    Examples:
        >>> a, b, c, s, n = sp.symbols("a b c s n")
        >>> expr = a * s**2 + b * s + c
        >>> completed_expr, substitution = complete_the_square(expr, symbol_to_complete=s, new_variable=n)
        >>> completed_expr
        a*n**2 + c - b**2/(4*a)
        >>> substitution
        (n, s + b/(2*a))

    Returns:
        (replaced_expr, tuple to pass to subs, such that old expr comes out again)
    """
    p = sp.Poly(expr, symbol_to_complete)
    coefficients = p.all_coeffs()
    if len(coefficients) != 3:
        return expr, None
    a, b, _ = coefficients
    expr = expr.subs(symbol_to_complete, new_variable - b / (2 * a))
    return sp.simplify(expr), (new_variable, symbol_to_complete + b / (2 * a))


def complete_the_squares_in_exp(expr: sp.Expr, symbols_to_complete: Sequence[sp.Symbol]):
    """Completes squares in arguments of exponential which makes them simpler to integrate.

    Very useful for integrating Maxwell-Boltzmann equilibria and its moment generating function
    """
    dummies = [sp.Dummy() for _ in symbols_to_complete]

    def visit(term):
        if term.func == sp.exp:
            exp_arg = term.args[0]
            for symbol_to_complete, dummy in zip(symbols_to_complete, dummies):
                exp_arg, substitution = complete_the_square(exp_arg, symbol_to_complete, dummy)
            return sp.exp(sp.expand(exp_arg))
        else:
            param_list = [visit(a) for a in term.args]
            if not param_list:
                return term
            else:
                return term.func(*param_list)

    result = visit(expr)
    for s, d in zip(symbols_to_complete, dummies):
        result = result.subs(d, s)
    return result


def extract_most_common_factor(term):
    """Processes a sum of fractions: determines the most common factor and splits term in common factor and rest"""
    coefficient_dict = term.as_coefficients_dict()
    counter = Counter([Abs(v) for v in coefficient_dict.values()])
    common_factor, occurrences = max(counter.items(), key=operator.itemgetter(1))
    if occurrences == 1 and (1 in counter):
        common_factor = 1
    return common_factor, term / common_factor


def recursive_collect(expr, symbols, order_by_occurences=False):
    """Applies sympy.collect recursively for a list of symbols, collecting symbol 2 in the coefficients of symbol 1,
    and so on.
    
    ``expr`` must be rewritable as a polynomial in the given ``symbols``.
    It it is not, ``recursive_collect`` will fail quietly, returning the original expression.

    Args:
        expr: A sympy expression.
        symbols: A sequence of symbols
        order_by_occurences: If True, during recursive descent, always collect the symbol occuring 
                             most often in the expression.
    """
    if order_by_occurences:
        symbols = list(expr.atoms(sp.Symbol) & set(symbols))
        symbols = sorted(symbols, key=expr.count, reverse=True)
    if len(symbols) == 0:
        return expr
    symbol = symbols[0]
    collected = expr.collect(symbol)
    
    try:
        collected_poly = sp.Poly(collected, symbol)
    except PolynomialError:
        return expr

    coeffs = collected_poly.all_coeffs()[::-1]
    rec_sum = sum(symbol**i * recursive_collect(c, symbols[1:], order_by_occurences) for i, c in enumerate(coeffs))
    return rec_sum


def summands(expr):
    return set(expr.args) if isinstance(expr, sp.Add) else {expr}


def simplify_by_equality(expr, a, b, c):
    """
    Uses the equality a = b + c, where a and b must be symbols, to simplify expr 
    by attempting to express additive combinations of two quantities by the third.

    This works on expressions that are reducible to the form 
    :math:`a * (...) + b * (...) + c * (...)`,
    without any mixed terms of a, b and c.
    """
    if not isinstance(a, sp.Symbol) or not isinstance(b, sp.Symbol):
        raise ValueError("a and b must be symbols.")

    c = sp.sympify(c)

    if not (isinstance(c, sp.Symbol) or is_constant(c)):
        raise ValueError("c must be either a symbol or a constant!")

    expr = sp.sympify(expr)

    expr_expanded = sp.expand(expr)
    a_coeff = expr_expanded.coeff(a, 1)
    expr_expanded -= (a * a_coeff).expand()
    b_coeff = expr_expanded.coeff(b, 1)
    expr_expanded -= (b * b_coeff).expand()
    if isinstance(c, sp.Symbol):
        c_coeff = expr_expanded.coeff(c, 1)
        rest = expr_expanded - (c * c_coeff).expand()
    else:
        c_coeff = expr_expanded / c
        rest = 0

    a_summands = summands(a_coeff)
    b_summands = summands(b_coeff)
    c_summands = summands(c_coeff)

    # replace b + c by a
    b_plus_c_coeffs = b_summands & c_summands
    for coeff in b_plus_c_coeffs:
        rest += a * coeff
    b_summands -= b_plus_c_coeffs
    c_summands -= b_plus_c_coeffs

    # replace a - b by c
    neg_b_summands = {-x for x in b_summands}
    a_minus_b_coeffs = a_summands & neg_b_summands
    for coeff in a_minus_b_coeffs:
        rest += c * coeff
    a_summands -= a_minus_b_coeffs
    b_summands -= {-x for x in a_minus_b_coeffs}

    # replace a - c by b
    neg_c_summands = {-x for x in c_summands}
    a_minus_c_coeffs = a_summands & neg_c_summands
    for coeff in a_minus_c_coeffs:
        rest += b * coeff
    a_summands -= a_minus_c_coeffs
    c_summands -= {-x for x in a_minus_c_coeffs}

    # put it back together
    return (rest + a * sum(a_summands) + b * sum(b_summands) + c * sum(c_summands)).expand()


def count_operations(term: Union[sp.Expr, List[sp.Expr], List[Assignment]],
                     only_type: Optional[str] = 'real') -> Dict[str, int]:
    """Counts the number of additions, multiplications and division.

    Args:
        term: a sympy expression (term, assignment) or sequence of sympy objects
        only_type: 'real' or 'int' to count only operations on these types, or None for all

    Returns:
        dict with 'adds', 'muls' and 'divs' keys
    """
    from pystencils.fast_approximation import fast_sqrt, fast_inv_sqrt, fast_division

    result = {'adds': 0, 'muls': 0, 'divs': 0, 'sqrts': 0,
              'fast_sqrts': 0, 'fast_inv_sqrts': 0, 'fast_div': 0}
    if isinstance(term, Sequence):
        for element in term:
            r = count_operations(element, only_type)
            for operation_name in result.keys():
                result[operation_name] += r[operation_name]
        return result
    elif isinstance(term, Assignment):
        term = term.rhs

    def check_type(e):
        if only_type is None:
            return True
        if isinstance(e, FieldPointerSymbol) and only_type == "real":
            return only_type == "int"

        try:
            base_type = get_type_of_expression(e)
        except ValueError:
            return False
        if isinstance(base_type, VectorType):
            return False
        if isinstance(base_type, PointerType):
            return only_type == 'int'
        if only_type == 'int' and (base_type.is_int() or base_type.is_uint()):
            return True
        if only_type == 'real' and (base_type.is_float()):
            return True
        else:
            return base_type == only_type

    def visit(t):
        visit_children = True
        if t.func is sp.Add:
            if check_type(t):
                result['adds'] += len(t.args) - 1
        elif t.func in [sp.Or, sp.And]:
            pass
        elif t.func is sp.Mul:
            if check_type(t):
                result['muls'] += len(t.args) - 1
                for a in t.args:
                    if a == 1 or a == -1:
                        result['muls'] -= 1
        elif isinstance(t, sp.Float) or isinstance(t, sp.Rational):
            pass
        elif isinstance(t, sp.Symbol):
            visit_children = False
        elif isinstance(t, sp.Indexed):
            visit_children = False
        elif t.is_integer:
            pass
        elif isinstance(t, CastFunc):
            visit_children = False
            visit(t.args[0])
        elif t.func is fast_sqrt:
            result['fast_sqrts'] += 1
        elif t.func is fast_inv_sqrt:
            result['fast_inv_sqrts'] += 1
        elif t.func is fast_division:
            result['fast_div'] += 1
        elif t.func is sp.Pow:
            if check_type(t.args[0]):
                visit_children = True
                if t.exp.is_integer and t.exp.is_number:
                    if t.exp >= 0:
                        result['muls'] += int(t.exp) - 1
                    else:
                        if result['muls'] > 0:
                            result['muls'] -= 1
                        result['divs'] += 1
                        result['muls'] += (-int(t.exp)) - 1
                elif sp.nsimplify(t.exp) == sp.Rational(1, 2):
                    result['sqrts'] += 1
                elif sp.nsimplify(t.exp) == -sp.Rational(1, 2):
                    result["sqrts"] += 1
                    result["divs"] += 1
                else:
                    warnings.warn(f"Cannot handle exponent {t.exp} of sp.Pow node")
            else:
                warnings.warn("Counting operations: only integer exponents are supported in Pow, "
                              "counting will be inaccurate")
        elif t.func is sp.Piecewise:
            for child_term, condition in t.args:
                visit(child_term)
            visit_children = False
        elif isinstance(t, (sp.Rel, sp.UnevaluatedExpr)):
            pass
        elif isinstance(t, DivFunc):
            result["divs"] += 1
        else:
            warnings.warn(f"Unknown sympy node of type {str(t.func)} counting will be inaccurate")

        if visit_children:
            for a in t.args:
                visit(a)

    visit(term)
    return result


def count_operations_in_ast(ast) -> Dict[str, int]:
    """Counts number of operations in an abstract syntax tree, see also :func:`count_operations`"""
    from pystencils.astnodes import SympyAssignment
    result = defaultdict(int)

    def visit(node):
        if isinstance(node, SympyAssignment):
            r = count_operations(node.rhs)
            for k, v in r.items():
                result[k] += v
        else:
            for arg in node.args:
                visit(arg)
    visit(ast)
    return result


def common_denominator(expr: sp.Expr) -> sp.Expr:
    """Finds least common multiple of all denominators occurring in an expression"""
    denominators = [r.q for r in expr.atoms(sp.Rational)]
    return sp.lcm(denominators)


def get_symmetric_part(expr: sp.Expr, symbols: Iterable[sp.Symbol]) -> sp.Expr:
    """
    Returns the symmetric part of a sympy expressions.

    Args:
        expr: sympy expression, labeled here as :math:`f`
        symbols: sequence of symbols which are considered as degrees of freedom, labeled here as :math:`x_0, x_1,...`

    Returns:
        :math:`\frac{1}{2} [ f(x_0, x_1, ..) + f(-x_0, -x_1) ]`
    """
    substitution_dict = {e: -e for e in symbols}
    return sp.Rational(1, 2) * (expr + expr.subs(substitution_dict))


class SymbolCreator:
    def __getattribute__(self, name):
        return sp.Symbol(name)
