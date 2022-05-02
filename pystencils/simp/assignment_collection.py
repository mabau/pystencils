import itertools
from copy import copy
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Union

import sympy as sp

import pystencils
from pystencils.assignment import Assignment
from pystencils.simp.simplifications import (sort_assignments_topologically, transform_lhs_and_rhs, transform_rhs)
from pystencils.sympyextensions import count_operations, fast_subs


class AssignmentCollection:
    """
    A collection of equations with subexpression definitions, also represented as assignments,
    that are used in the main equations. AssignmentCollection can be passed to simplification methods.
    These simplification methods can change the subexpressions, but the number and
    left hand side of the main equations themselves is not altered.
    Additionally a dictionary of simplification hints is stored, which are set by the functions that create
    assignment collections to transport information to the simplification system.

    Args:
        main_assignments: List of assignments. Main assignments are characterised, that the right hand side of each
                          assignment is a field access. Thus the generated equations write on arrays.
        subexpressions: List of assignments defining subexpressions used in main equations
        simplification_hints: Dict that is used to annotate the assignment collection with hints that are
                              used by the simplification system. See documentation of the simplification rules for
                              potentially required hints and their meaning.
        subexpression_symbol_generator: Generator for new symbols that are used when new subexpressions are added
                                        used to get new symbols that are unique for this AssignmentCollection

    """

    # ------------------------------- Creation & Inplace Manipulation --------------------------------------------------

    def __init__(self, main_assignments: Union[List[Assignment], Dict[sp.Expr, sp.Expr]],
                 subexpressions: Union[List[Assignment], Dict[sp.Expr, sp.Expr]] = None,
                 simplification_hints: Optional[Dict[str, Any]] = None,
                 subexpression_symbol_generator: Iterator[sp.Symbol] = None) -> None:

        if subexpressions is None:
            subexpressions = {}

        if isinstance(main_assignments, Dict):
            main_assignments = [Assignment(k, v)
                                for k, v in main_assignments.items()]
        if isinstance(subexpressions, Dict):
            subexpressions = [Assignment(k, v)
                              for k, v in subexpressions.items()]

        main_assignments = list(itertools.chain.from_iterable(
            [(a if isinstance(a, Iterable) else [a]) for a in main_assignments]))
        subexpressions = list(itertools.chain.from_iterable(
            [(a if isinstance(a, Iterable) else [a]) for a in subexpressions]))

        self.main_assignments = main_assignments
        self.subexpressions = subexpressions

        if simplification_hints is None:
            simplification_hints = {}

        self.simplification_hints = simplification_hints

        if subexpression_symbol_generator is None:
            self.subexpression_symbol_generator = SymbolGen()
        else:
            self.subexpression_symbol_generator = subexpression_symbol_generator

    def add_simplification_hint(self, key: str, value: Any) -> None:
        """Adds an entry to the simplification_hints dictionary and checks that is does not exist yet."""
        assert key not in self.simplification_hints, "This hint already exists"
        self.simplification_hints[key] = value

    def add_subexpression(self, rhs: sp.Expr, lhs: Optional[sp.Symbol] = None, topological_sort=True) -> sp.Symbol:
        """Adds a subexpression to current collection.

        Args:
            rhs: right hand side of new subexpression
            lhs: optional left hand side of new subexpression. If None a new unique symbol is generated.
            topological_sort: sort the subexpressions topologically after insertion, to make sure that
                              definition of a symbol comes before its usage. If False, subexpression is appended.

        Returns:
            left hand side symbol (which could have been generated)
        """
        if lhs is None:
            lhs = next(self.subexpression_symbol_generator)
        eq = Assignment(lhs, rhs)
        self.subexpressions.append(eq)
        if topological_sort:
            self.topological_sort(sort_subexpressions=True,
                                  sort_main_assignments=False)
        return lhs

    def topological_sort(self, sort_subexpressions: bool = True, sort_main_assignments: bool = True) -> None:
        """Sorts subexpressions and/or main_equations topologically to make sure symbol usage comes after definition."""
        if sort_subexpressions:
            self.subexpressions = sort_assignments_topologically(self.subexpressions)
        if sort_main_assignments:
            self.main_assignments = sort_assignments_topologically(self.main_assignments)

    # ---------------------------------------------- Properties  -------------------------------------------------------

    @property
    def all_assignments(self) -> List[Assignment]:
        """Subexpression and main equations as a single list."""
        return self.subexpressions + self.main_assignments

    @property
    def rhs_symbols(self) -> Set[sp.Symbol]:
        """All symbols used in the assignment collection, which occur on the rhs of any assignment."""
        rhs_symbols = set()
        for eq in self.all_assignments:
            if isinstance(eq, Assignment):
                rhs_symbols.update(eq.rhs.atoms(sp.Symbol))
            elif isinstance(eq, pystencils.astnodes.Node):
                rhs_symbols.update(eq.undefined_symbols)

        return rhs_symbols

    @property
    def free_symbols(self) -> Set[sp.Symbol]:
        """All symbols used in the assignment collection, which do not occur as left hand sides in any assignment."""
        return self.rhs_symbols - self.bound_symbols

    @property
    def bound_symbols(self) -> Set[sp.Symbol]:
        """All symbols which occur on the left hand side of a main assignment or a subexpression."""
        bound_symbols_set = set(
            [assignment.lhs for assignment in self.all_assignments if isinstance(assignment, Assignment)]
        )

        assert len(bound_symbols_set) == len(list(a for a in self.all_assignments if isinstance(a, Assignment))), \
            "Not in SSA form - same symbol assigned multiple times"

        bound_symbols_set = bound_symbols_set.union(*[
            assignment.symbols_defined for assignment in self.all_assignments
            if isinstance(assignment, pystencils.astnodes.Node)
        ])

        return bound_symbols_set

    @property
    def rhs_fields(self):
        """All fields accessed in the assignment collection, which do not occur as left hand sides in any assignment."""
        return {s.field for s in self.rhs_symbols if hasattr(s, 'field')}

    @property
    def free_fields(self):
        """All fields accessed in the assignment collection, which do not occur as left hand sides in any assignment."""
        return {s.field for s in self.free_symbols if hasattr(s, 'field')}

    @property
    def bound_fields(self):
        """All field accessed on the left hand side of a main assignment or a subexpression."""
        return {s.field for s in self.bound_symbols if hasattr(s, 'field')}

    @property
    def defined_symbols(self) -> Set[sp.Symbol]:
        """All symbols which occur as left-hand-sides of one of the main equations"""
        lhs_set = set([assignment.lhs for assignment in self.main_assignments if isinstance(assignment, Assignment)])
        return (lhs_set.union(*[assignment.symbols_defined for assignment in self.main_assignments
                                if isinstance(assignment, pystencils.astnodes.Node)]))

    @property
    def operation_count(self):
        """See :func:`count_operations` """
        return count_operations(self.all_assignments, only_type=None)

    def atoms(self, *args):
        return set().union(*[a.atoms(*args) for a in self.all_assignments])

    def dependent_symbols(self, symbols: Iterable[sp.Symbol]) -> Set[sp.Symbol]:
        """Returns all symbols that depend on one of the passed symbols.

        A symbol 'a' depends on a symbol 'b', if there is an assignment 'a <- some_expression(b)' i.e. when
        'b' is required to compute 'a'.
        """

        queue = list(symbols)

        def add_symbols_from_expr(expr):
            dependent_symbols = expr.atoms(sp.Symbol)
            for ds in dependent_symbols:
                queue.append(ds)

        handled_symbols = set()
        assignment_dict = {e.lhs: e.rhs for e in self.all_assignments}

        while len(queue) > 0:
            e = queue.pop(0)
            if e in handled_symbols:
                continue
            if e in assignment_dict:
                add_symbols_from_expr(assignment_dict[e])
            handled_symbols.add(e)

        return handled_symbols

    def lambdify(self, symbols: Sequence[sp.Symbol], fixed_symbols: Optional[Dict[sp.Symbol, Any]] = None, module=None):
        """Returns a python function to evaluate this equation collection.

        Args:
            symbols: symbol(s) which are the parameter for the created function
            fixed_symbols: dictionary with substitutions, that are applied before sympy's lambdify
            module: same as sympy.lambdify parameter. Defines which module to use e.g. 'numpy'

        Examples:
              >>> a, b, c, d = sp.symbols("a b c d")
              >>> ac = AssignmentCollection([Assignment(c, a + b), Assignment(d, a**2 + b)],
              ...                           subexpressions=[Assignment(b, a + b / 2)])
              >>> python_function = ac.lambdify([a], fixed_symbols={b: 2})
              >>> python_function(4)
              {c: 6, d: 18}
        """
        assignments = self.new_with_substitutions(fixed_symbols, substitute_on_lhs=False) if fixed_symbols else self
        assignments = assignments.new_without_subexpressions().main_assignments
        lambdas = {assignment.lhs: sp.lambdify(symbols, assignment.rhs, module) for assignment in assignments}

        def f(*args, **kwargs):
            return {s: func(*args, **kwargs) for s, func in lambdas.items()}

        return f

    # ---------------------------- Creating new modified collections ---------------------------------------------------

    def copy(self,
             main_assignments: Optional[List[Assignment]] = None,
             subexpressions: Optional[List[Assignment]] = None) -> 'AssignmentCollection':
        """Returns a copy with optionally replaced main_assignments and/or subexpressions."""

        res = copy(self)
        res.simplification_hints = self.simplification_hints.copy()
        res.subexpression_symbol_generator = copy(self.subexpression_symbol_generator)

        if main_assignments is not None:
            res.main_assignments = main_assignments
        else:
            res.main_assignments = self.main_assignments.copy()

        if subexpressions is not None:
            res.subexpressions = subexpressions
        else:
            res.subexpressions = self.subexpressions.copy()

        return res

    def new_with_substitutions(self, substitutions: Dict, add_substitutions_as_subexpressions: bool = False,
                               substitute_on_lhs: bool = True,
                               sort_topologically: bool = True) -> 'AssignmentCollection':
        """Returns new object, where terms are substituted according to the passed substitution dict.

        Args:
            substitutions: dict that is passed to sympy subs, substitutions are done main assignments and subexpressions
            add_substitutions_as_subexpressions: if True, the substitutions are added as assignments to subexpressions
            substitute_on_lhs: if False, the substitutions are done only on the right hand side of assignments
            sort_topologically: if subexpressions are added as substitutions and this parameters is true,
                                the subexpressions are sorted topologically after insertion
        Returns:
            New AssignmentCollection where substitutions have been applied, self is not altered.
        """
        transform = transform_lhs_and_rhs if substitute_on_lhs else transform_rhs
        transformed_subexpressions = transform(self.subexpressions, fast_subs, substitutions)
        transformed_assignments = transform(self.main_assignments, fast_subs, substitutions)

        if add_substitutions_as_subexpressions:
            transformed_subexpressions = [Assignment(b, a) for a, b in
                                          substitutions.items()] + transformed_subexpressions
            if sort_topologically:
                transformed_subexpressions = sort_assignments_topologically(transformed_subexpressions)
        return self.copy(transformed_assignments, transformed_subexpressions)

    def new_merged(self, other: 'AssignmentCollection') -> 'AssignmentCollection':
        """Returns a new collection which contains self and other. Subexpressions are renamed if they clash."""
        own_definitions = set([e.lhs for e in self.main_assignments])
        other_definitions = set([e.lhs for e in other.main_assignments])
        assert len(own_definitions.intersection(other_definitions)) == 0, \
            "Cannot merge collections, since both define the same symbols"

        own_subexpression_symbols = {e.lhs: e.rhs for e in self.subexpressions}
        substitution_dict = {}

        processed_other_subexpression_equations = []
        for other_subexpression_eq in other.subexpressions:
            if other_subexpression_eq.lhs in own_subexpression_symbols:
                if other_subexpression_eq.rhs == own_subexpression_symbols[other_subexpression_eq.lhs]:
                    continue  # exact the same subexpression equation exists already
                else:
                    # different definition - a new name has to be introduced
                    new_lhs = next(self.subexpression_symbol_generator)
                    new_eq = Assignment(new_lhs, fast_subs(other_subexpression_eq.rhs, substitution_dict))
                    processed_other_subexpression_equations.append(new_eq)
                    substitution_dict[other_subexpression_eq.lhs] = new_lhs
            else:
                processed_other_subexpression_equations.append(fast_subs(other_subexpression_eq, substitution_dict))

        processed_other_main_assignments = [fast_subs(eq, substitution_dict) for eq in other.main_assignments]
        return self.copy(self.main_assignments + processed_other_main_assignments,
                         self.subexpressions + processed_other_subexpression_equations)

    def new_filtered(self, symbols_to_extract: Iterable[sp.Symbol]) -> 'AssignmentCollection':
        """Extracts equations that have symbols_to_extract as left hand side, together with necessary subexpressions.

        Returns:
            new AssignmentCollection, self is not altered
        """
        symbols_to_extract = set(symbols_to_extract)
        dependent_symbols = self.dependent_symbols(symbols_to_extract)
        new_assignments = []
        for eq in self.all_assignments:
            if eq.lhs in symbols_to_extract:
                new_assignments.append(eq)

        new_sub_expr = [eq for eq in self.all_assignments
                        if eq.lhs in dependent_symbols and eq.lhs not in symbols_to_extract]
        return self.copy(new_assignments, new_sub_expr)

    def new_without_unused_subexpressions(self) -> 'AssignmentCollection':
        """Returns new collection that only contains subexpressions required to compute the main assignments."""
        all_lhs = [eq.lhs for eq in self.main_assignments]
        return self.new_filtered(all_lhs)

    def new_with_inserted_subexpression(self, symbol: sp.Symbol) -> 'AssignmentCollection':
        """Eliminates the subexpression with the given symbol on its left hand side, by substituting it everywhere."""
        new_subexpressions = []
        subs_dict = None
        for se in self.subexpressions:
            if se.lhs == symbol:
                subs_dict = {se.lhs: se.rhs}
            else:
                new_subexpressions.append(se)
        if subs_dict is None:
            return self

        new_subexpressions = [Assignment(eq.lhs, fast_subs(eq.rhs, subs_dict)) for eq in new_subexpressions]
        new_eqs = [Assignment(eq.lhs, fast_subs(eq.rhs, subs_dict)) for eq in self.main_assignments]
        return self.copy(new_eqs, new_subexpressions)

    def new_without_subexpressions(self, subexpressions_to_keep=None) -> 'AssignmentCollection':
        """Returns a new collection where all subexpressions have been inserted."""
        if subexpressions_to_keep is None:
            subexpressions_to_keep = set()
        if len(self.subexpressions) == 0:
            return self.copy()

        subexpressions_to_keep = set(subexpressions_to_keep)

        kept_subexpressions = []
        if self.subexpressions[0].lhs in subexpressions_to_keep:
            substitution_dict = {}
            kept_subexpressions.append(self.subexpressions[0])
        else:
            substitution_dict = {self.subexpressions[0].lhs: self.subexpressions[0].rhs}

        subexpression = [e for e in self.subexpressions]
        for i in range(1, len(subexpression)):
            subexpression[i] = fast_subs(subexpression[i], substitution_dict)
            if subexpression[i].lhs in subexpressions_to_keep:
                kept_subexpressions.append(subexpression[i])
            else:
                substitution_dict[subexpression[i].lhs] = subexpression[i].rhs

        new_assignment = [fast_subs(eq, substitution_dict) for eq in self.main_assignments]
        return self.copy(new_assignment, kept_subexpressions)

    # ----------------------------------------- Display and Printing   -------------------------------------------------

    def _repr_html_(self):
        """Interface to Jupyter notebook, to display as a nicely formatted HTML table"""

        def make_html_equation_table(equations):
            no_border = 'style="border:none"'
            html_table = '<table style="border:none; width: 100%; ">'
            line = '<tr {nb}> <td {nb}>$${eq}$$</td>  </tr> '
            for eq in equations:
                format_dict = {'eq': sp.latex(eq),
                               'nb': no_border, }
                html_table += line.format(**format_dict)
            html_table += "</table>"
            return html_table

        result = ""
        if len(self.subexpressions) > 0:
            result += "<div>Subexpressions:</div>"
            result += make_html_equation_table(self.subexpressions)
        result += "<div>Main Assignments:</div>"
        result += make_html_equation_table(self.main_assignments)
        return result

    def __repr__(self):
        return f"AssignmentCollection: {str(tuple(self.defined_symbols))[1:-1]} <- f{tuple(self.free_symbols)}"

    def __str__(self):
        result = "Subexpressions:\n"
        for eq in self.subexpressions:
            result += f"\t{eq}\n"
        result += "Main Assignments:\n"
        for eq in self.main_assignments:
            result += f"\t{eq}\n"
        return result

    def __iter__(self):
        return self.all_assignments.__iter__()

    @property
    def main_assignments_dict(self):
        return {a.lhs: a.rhs for a in self.main_assignments}

    @property
    def subexpressions_dict(self):
        return {a.lhs: a.rhs for a in self.subexpressions}

    def set_main_assignments_from_dict(self, main_assignments_dict):
        self.main_assignments = [Assignment(k, v)
                                 for k, v in main_assignments_dict.items()]

    def set_sub_expressions_from_dict(self, sub_expressions_dict):
        self.subexpressions = [Assignment(k, v)
                               for k, v in sub_expressions_dict.items()]

    def find(self, *args, **kwargs):
        return set.union(
            *[a.find(*args, **kwargs) for a in self.all_assignments]
        )

    def match(self, *args, **kwargs):
        rtn = {}
        for a in self.all_assignments:
            partial_result = a.match(*args, **kwargs)
            if partial_result:
                rtn.update(partial_result)
        return rtn

    def subs(self, *args, **kwargs):
        return AssignmentCollection(
            main_assignments=[a.subs(*args, **kwargs) for a in self.main_assignments],
            subexpressions=[a.subs(*args, **kwargs) for a in self.subexpressions]
        )

    def replace(self, *args, **kwargs):
        return AssignmentCollection(
            main_assignments=[a.replace(*args, **kwargs) for a in self.main_assignments],
            subexpressions=[a.replace(*args, **kwargs) for a in self.subexpressions]
        )

    def __eq__(self, other):
        return set(self.all_assignments) == set(other.all_assignments)

    def __bool__(self):
        return bool(self.all_assignments)


class SymbolGen:
    """Default symbol generator producing number symbols ζ_0, ζ_1, ..."""

    def __init__(self, symbol="xi", dtype=None):
        self._ctr = 0
        self._symbol = symbol
        self._dtype = dtype

    def __iter__(self):
        return self

    def __next__(self):
        name = f"{self._symbol}_{self._ctr}"
        self._ctr += 1
        if self._dtype is not None:
            return pystencils.TypedSymbol(name, self._dtype)
        return sp.Symbol(name)
