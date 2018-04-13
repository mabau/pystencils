import sympy as sp
from collections import namedtuple, defaultdict
from pystencils.sympyextensions import normalize_product, prod


def default_diff_sort_key(d):
    return str(d.superscript), str(d.target)


class Diff(sp.Expr):
    """
    Sympy Node representing a derivative. The difference to sympy's built in differential is:
        - shortened latex representation
        - all simplifications have to be done manually
        - optional marker displayed as superscript
    """
    is_number = False
    is_Rational = False
    _diff_wrt = True

    def __new__(cls, argument, target=-1, superscript=-1):
        if argument == 0:
            return sp.Rational(0, 1)
        return sp.Expr.__new__(cls, argument.expand(), sp.sympify(target), sp.sympify(superscript))

    @property
    def is_commutative(self):
        any_non_commutative = any(not s.is_commutative for s in self.atoms(sp.Symbol))
        if any_non_commutative:
            return False
        else:
            return True

    def get_arg_recursive(self):
        """Returns the argument the derivative acts on, for nested derivatives the inner argument is returned"""
        if not isinstance(self.arg, Diff):
            return self.arg
        else:
            return self.arg.get_arg_recursive()

    def change_arg_recursive(self, new_arg):
        """Returns a Diff node with the given 'new_arg' instead of the current argument. For nested derivatives
        a new nested derivative is returned where the inner Diff has the 'new_arg'"""
        if not isinstance(self.arg, Diff):
            return Diff(new_arg, self.target, self.superscript)
        else:
            return Diff(self.arg.change_arg_recursive(new_arg), self.target, self.superscript)

    def split_linear(self, functions):
        """
        Applies linearity property of Diff: i.e.  'Diff(c*a+b)' is transformed to 'c * Diff(a) + Diff(b)'
        The parameter functions is a list of all symbols that are considered functions, not constants.
        For the example above: functions=[a, b]
        """
        constant, variable = 1, 1

        if self.arg.func != sp.Mul:
            constant, variable = 1, self.arg
        else:
            for factor in normalize_product(self.arg):
                if factor in functions or isinstance(factor, Diff):
                    variable *= factor
                else:
                    constant *= factor

        if isinstance(variable, sp.Symbol) and variable not in functions:
            return 0

        if isinstance(variable, int) or variable.is_number:
            return 0
        else:
            return constant * Diff(variable, target=self.target, superscript=self.superscript)

    @property
    def arg(self):
        """Expression the derivative acts on"""
        return self.args[0]

    @property
    def target(self):
        """Subscript, usually the variable the Diff is w.r.t. """
        return self.args[1]

    @property
    def superscript(self):
        """Superscript, for example used as the Chapman-Enskog order index"""
        return self.args[2]

    def _latex(self, printer, *_):
        result = "{\partial"
        if self.superscript >= 0:
            result += "^{(%s)}" % (self.superscript,)
        if self.target != -1:
            result += "_{%s}" % (printer.doprint(self.target),)

        contents = printer.doprint(self.arg)
        if isinstance(self.arg, int) or isinstance(self.arg, sp.Symbol) or self.arg.is_number or self.arg.func == Diff:
            result += " " + contents
        else:
            result += " (" + contents + ") "

        result += "}"
        return result

    def __str__(self):
        return "D(%s)" % self.arg


class DiffOperator(sp.Expr):
    """
    Un-applied differential, i.e. differential operator
    Its args are:
        - target: the differential is w.r.t to this variable.
                 This target is mainly for display purposes (its the subscript) and to distinguish DiffOperators
                 If the target is '-1' no subscript is displayed
        - superscript: optional marker displayed as superscript
                        is not displayed if set to '-1'
    The DiffOperator behaves much like a variable with special name. Its main use is to be applied later, using the
    DiffOperator.apply(expr, arg) which transforms 'DiffOperator's to applied 'Diff's
    """
    is_commutative = True
    is_number = False
    is_Rational = False

    def __new__(cls, target=-1, superscript=-1):
        return sp.Expr.__new__(cls, sp.sympify(target), sp.sympify(superscript))

    @property
    def target(self):
        return self.args[0]

    @property
    def superscript(self):
        return self.args[1]

    def _latex(self, *_):
        result = "{\partial"
        if self.superscript >= 0:
            result += "^{(%s)}" % (self.superscript,)
        if self.target != -1:
            result += "_{%s}" % (self.target,)
        result += "}"
        return result

    @staticmethod
    def apply(expr, argument, apply_to_constants=True):
        """
        Returns a new expression where each 'DiffOperator' is replaced by a 'Diff' node.
        Multiplications of 'DiffOperator's are interpreted as nested application of differentiation:
        i.e. DiffOperator('x')*DiffOperator('x') is a second derivative replaced by Diff(Diff(arg, x), t)
        """

        def handle_mul(mul):
            args = normalize_product(mul)
            diffs = [a for a in args if isinstance(a, DiffOperator)]
            if len(diffs) == 0:
                return mul * argument if apply_to_constants else mul
            rest = [a for a in args if not isinstance(a, DiffOperator)]
            diffs.sort(key=default_diff_sort_key)
            result = argument
            for d in reversed(diffs):
                result = Diff(result, target=d.target, superscript=d.superscript)
            return prod(rest) * result

        expr = expr.expand()
        if expr.func == sp.Mul or expr.func == sp.Pow:
            return handle_mul(expr)
        elif expr.func == sp.Add:
            return expr.func(*[handle_mul(a) for a in expr.args])
        else:
            return expr * argument if apply_to_constants else expr


# ----------------------------------------------------------------------------------------------------------------------


def derivative_terms(expr):
    """
    Returns set of all derivatives in an expression
    this is different from `expr.atoms(Diff)` when nested derivatives are in the expression,
    since this function only returns the outer derivatives
    """
    result = set()

    def visit(e):
        if isinstance(e, Diff):
            result.add(e)
        else:
            for a in e.args:
                visit(a)

    visit(expr)
    return result


def collect_derivatives(expr):
    """Rewrites expression into a sum of distinct derivatives with pre-factors"""
    return expr.collect(derivative_terms(expr))


def create_nested_diff(arg, *args):
    """Shortcut to create nested derivatives"""
    assert arg is not None
    args = sorted(args, reverse=True, key=lambda e: e.name if isinstance(e, sp.Symbol) else e)
    res = arg
    for i in args:
        res = Diff(res, i)
    return res


def expand_using_linearity(expr, functions=None, constants=None):
    """
    Expands all derivative nodes by applying Diff.split_linear
    :param expr: expression containing derivatives
    :param functions: sequence of symbols that are considered functions and can not be pulled before the derivative.
                      if None, all symbols are viewed as functions
    :param constants: sequence of symbols which are considered constants and can be pulled before the derivative
    """
    if functions is None:
        functions = expr.atoms(sp.Symbol)
        if constants is not None:
            functions.difference_update(constants)

    if isinstance(expr, Diff):
        arg = expand_using_linearity(expr.arg, functions)
        if hasattr(arg, 'func') and arg.func == sp.Add:
            result = 0
            for a in arg.args:
                result += Diff(a, target=expr.target, superscript=expr.superscript).split_linear(functions)
            return result
        else:
            diff = Diff(arg, target=expr.target, superscript=expr.superscript)
            if diff == 0:
                return 0
            else:
                return diff.split_linear(functions)
    else:
        new_args = [expand_using_linearity(e, functions) for e in expr.args]
        result = sp.expand(expr.func(*new_args) if new_args else expr)
        return result


def full_diff_expand(expr, functions=None, constants=None):
    if functions is None:
        functions = expr.atoms(sp.Symbol)
        if constants is not None:
            functions.difference_update(constants)

    def visit(e):
        e = e.expand()

        if e.func == Diff:
            result = 0
            diff_args = {'target': e.target, 'superscript': e.superscript}
            diff_inner = e.args[0]
            diff_inner = visit(diff_inner)
            for term in diff_inner.args if diff_inner.func == sp.Add else [diff_inner]:
                independent_terms = 1
                dependent_terms = []
                for factor in normalize_product(term):
                    if factor in functions or isinstance(factor, Diff):
                        dependent_terms.append(factor)
                    else:
                        independent_terms *= factor
                for i in range(len(dependent_terms)):
                    dependent_term = dependent_terms[i]
                    other_dependent_terms = dependent_terms[:i] + dependent_terms[i + 1:]
                    processed_diff = normalize_diff_order(Diff(dependent_term, **diff_args))
                    result += independent_terms * prod(other_dependent_terms) * processed_diff
            return result
        else:
            new_args = [visit(arg) for arg in e.args]
            return e.func(*new_args) if new_args else e

    if isinstance(expr, sp.Matrix):
        return expr.applyfunc(visit)
    else:
        return visit(expr)


def normalize_diff_order(expression, functions=None, constants=None, sort_key=default_diff_sort_key):
    """Assumes order of differentiation can be exchanged. Changes the order of nested Diffs to a standard order defined
    by the sorting key 'sort_key' such that the derivative terms can be further simplified """

    def visit(expr):
        if isinstance(expr, Diff):
            nodes = [expr]
            while isinstance(nodes[-1].arg, Diff):
                nodes.append(nodes[-1].arg)

            processed_arg = visit(nodes[-1].arg)
            nodes.sort(key=sort_key)

            result = processed_arg
            for d in reversed(nodes):
                result = Diff(result, target=d.target, superscript=d.superscript)
            return result
        else:
            new_args = [visit(e) for e in expr.args]
            return expr.func(*new_args) if new_args else expr

    expression = expand_using_linearity(expression.expand(), functions, constants).expand()
    return visit(expression)


def expand_using_product_rule(expr):
    """Fully expands all derivatives by applying product rule"""
    if isinstance(expr, Diff):
        arg = expand_using_product_rule(expr.args[0])
        if arg.func == sp.Add:
            new_args = [Diff(e, target=expr.target, superscript=expr.superscript)
                        for e in arg.args]
            return sp.Add(*new_args)
        if arg.func not in (sp.Mul, sp.Pow):
            return Diff(arg, target=expr.target, superscript=expr.superscript)
        else:
            prod_list = normalize_product(arg)
            result = 0
            for i in range(len(prod_list)):
                pre_factor = prod(prod_list[j] for j in range(len(prod_list)) if i != j)
                result += pre_factor * Diff(prod_list[i], target=expr.target, superscript=expr.superscript)
            return result
    else:
        new_args = [expand_using_product_rule(e) for e in expr.args]
        return expr.func(*new_args) if new_args else expr


def combine_using_product_rule(expr):
    """Inverse product rule"""

    def expr_to_diff_decomposition(expression):
        """Decomposes a sp.Add node containing CeDiffs into:
        diff_dict: maps (target, superscript) -> [ (pre_factor, argument), ... ]
        i.e.  a partial(b) ( a is pre-factor, b is argument)
            in case of partial(a) partial(b) two entries are created  (0.5 partial(a), b), (0.5 partial(b), a)
        """
        DiffInfo = namedtuple("DiffInfo", ["target", "superscript"])

        class DiffSplit:
            def __init__(self, fac, argument):
                self.pre_factor = fac
                self.argument = argument

            def __repr__(self):
                return str((self.pre_factor, self.argument))

        assert isinstance(expression, sp.Add)
        diff_dict = defaultdict(list)
        rest = 0
        for term in expression.args:
            if isinstance(term, Diff):
                diff_dict[DiffInfo(term.target, term.superscript)].append(DiffSplit(1, term.arg))
            else:
                mul_args = normalize_product(term)
                diffs = [d for d in mul_args if isinstance(d, Diff)]
                factor = prod(d for d in mul_args if not isinstance(d, Diff))
                if len(diffs) == 0:
                    rest += factor
                else:
                    for i, diff in enumerate(diffs):
                        all_but_current = [d for j, d in enumerate(diffs) if i != j]
                        pre_factor = factor * prod(all_but_current) * sp.Rational(1, len(diffs))
                        diff_dict[DiffInfo(diff.target, diff.superscript)].append(DiffSplit(pre_factor, diff.arg))

        return diff_dict, rest

    def match_diff_splits(own, other):
        own_fac = own.pre_factor / other.argument
        other_fac = other.pre_factor / own.argument
        count = sp.count_ops
        if count(own_fac) > count(own.pre_factor) or count(other_fac) > count(other.pre_factor):
            return None

        new_other_factor = own_fac - other_fac
        return new_other_factor

    def process_diff_list(diff_list, label, superscript):
        if len(diff_list) == 0:
            return 0
        elif len(diff_list) == 1:
            return diff_list[0].pre_factor * Diff(diff_list[0].argument, label, superscript)

        result = 0
        matches = []
        for i in range(1, len(diff_list)):
            match_result = match_diff_splits(diff_list[i], diff_list[0])
            if match_result is not None:
                matches.append((i, match_result))

        if len(matches) == 0:
            result += diff_list[0].pre_factor * Diff(diff_list[0].argument, label, superscript)
        else:
            other_idx, match_result = sorted(matches, key=lambda e: sp.count_ops(e[1]))[0]
            new_argument = diff_list[0].argument * diff_list[other_idx].argument
            result += (diff_list[0].pre_factor / diff_list[other_idx].argument) * Diff(new_argument, label, superscript)
            if match_result == 0:
                del diff_list[other_idx]
            else:
                diff_list[other_idx].pre_factor = match_result * diff_list[0].argument
        result += process_diff_list(diff_list[1:], label, superscript)
        return result

    def combine(expression):
        expression = expression.expand()
        if isinstance(expression, sp.Add):
            diff_dict, rest = expr_to_diff_decomposition(expression)
            for (label, superscript), diff_list in diff_dict.items():
                rest += process_diff_list(diff_list, label, superscript)
            return rest
        else:
            new_args = [combine_using_product_rule(e) for e in expression.args]
            return expression.func(*new_args) if new_args else expression

    return combine(expr)


def replace_diff(expr, replacement_dict):
    """replacement_dict: maps variable (target) to a new Differential operator"""

    def visit(e):
        if isinstance(e, Diff):
            if e.target in replacement_dict:
                return DiffOperator.apply(replacement_dict[e.target], visit(e.arg))
        new_args = [visit(arg) for arg in e.args]
        return e.func(*new_args) if new_args else e

    return visit(expr)


def zero_diffs(expr, label):
    """Replaces all differentials with the given target by 0"""

    def visit(e):
        if isinstance(e, Diff):
            if e.target == label:
                return 0
        new_args = [visit(arg) for arg in e.args]
        return e.func(*new_args) if new_args else e

    return visit(expr)


def evaluate_diffs(expr, var=None):
    """Replaces Diff nodes by sp.diff , the free variable is either the target (if var=None) otherwise
    the specified var"""
    if isinstance(expr, Diff):
        if var is None:
            var = expr.target
        return sp.diff(evaluate_diffs(expr.arg, var), var)
    else:
        new_args = [evaluate_diffs(arg, var) for arg in expr.args]
        return expr.func(*new_args) if new_args else expr


def functional_derivative(functional, v):
    r"""
    Computes functional derivative of functional with respect to v using Euler-Lagrange equation

    .. math ::

        \frac{\delta F}{\delta v} =
                \frac{\partial F}{\partial v} - \nabla \cdot \frac{\partial F}{\partial \nabla v}

    - assumes that gradients are represented by Diff() node
    - Diff(Diff(r)) represents the divergence of r
    - the constants parameter is a list with symbols not affected by the derivative. This is used for simplification
      of the derivative terms.
    """
    diffs = functional.atoms(Diff)
    bulk_substitutions = {d: sp.Dummy() for d in diffs}
    bulk_substitutions_inverse = {v: k for k, v in bulk_substitutions.items()}
    non_diff_part = functional.subs(bulk_substitutions)
    partial_f_partial_v = sp.diff(non_diff_part, v).subs(bulk_substitutions_inverse)

    gradient_part = 0
    for diff_obj in diffs:
        if diff_obj.args[0] != v:
            continue
        dummy = sp.Dummy()
        partial_f_partial_grad_v = functional.subs(diff_obj, dummy).diff(dummy).subs(dummy, diff_obj)
        gradient_part += Diff(partial_f_partial_grad_v, target=diff_obj.target, superscript=diff_obj.superscript)

    result = partial_f_partial_v - gradient_part
    return result
