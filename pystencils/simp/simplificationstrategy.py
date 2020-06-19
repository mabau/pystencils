from collections import namedtuple
from typing import Any, Callable, Optional, Sequence

import sympy as sp

from pystencils.simp.assignment_collection import AssignmentCollection


class SimplificationStrategy:
    """A simplification strategy is an ordered collection of simplification rules.

    Each simplification is a function taking an assignment collection, and returning a new simplified
    assignment collection. The strategy can nicely print intermediate simplification stages and results
    to Jupyter notebooks.
    """

    def __init__(self):
        self._rules = []

    def add(self, rule: Callable[[AssignmentCollection], AssignmentCollection]) -> None:
        """Adds the given simplification rule to the end of the collection.

        Args:
            rule: function that rewrites/simplifies an assignment collection
        """
        self._rules.append(rule)

    @property
    def rules(self):
        return self._rules

    def apply(self, assignment_collection: AssignmentCollection) -> AssignmentCollection:
        """Runs all rules on the given assignment collection."""
        for t in self._rules:
            assignment_collection = t(assignment_collection)
        return assignment_collection

    def __call__(self, assignment_collection: AssignmentCollection) -> AssignmentCollection:
        """Same as apply"""
        return self.apply(assignment_collection)

    def create_simplification_report(self, assignment_collection: AssignmentCollection) -> Any:
        """Creates a report to be displayed as HTML in a Jupyter notebook.

        The simplification report contains the number of operations at each simplification stage together
        with the run-time the simplification took.
        """

        ReportElement = namedtuple('ReportElement', ['simplificationName', 'runtime', 'adds', 'muls', 'divs', 'total'])

        class Report:
            def __init__(self):
                self.elements = []

            def add(self, element):
                self.elements.append(element)

            def __str__(self):
                try:
                    import tabulate
                    return tabulate(self.elements, headers=['Name', 'Runtime', 'Adds', 'Muls', 'Divs', 'Total'])
                except ImportError:
                    result = "Name, Adds, Muls, Divs, Runtime\n"
                    for e in self.elements:
                        result += ",".join([str(tuple_item) for tuple_item in e]) + "\n"
                    return result

            def _repr_html_(self):
                html_table = '<table style="border:none">'
                html_table += "<tr><th>Name</th>" \
                              "<th>Runtime</th>" \
                              "<th>Adds</th>" \
                              "<th>Muls</th>" \
                              "<th>Divs</th>" \
                              "<th>Total</th></tr>"
                line = "<tr><td>{simplificationName}</td>" \
                       "<td>{runtime}</td> <td>{adds}</td> <td>{muls}</td> <td>{divs}</td>  <td>{total}</td> </tr>"

                for e in self.elements:
                    # noinspection PyProtectedMember
                    html_table += line.format(**e._asdict())
                html_table += "</table>"
                return html_table

        import timeit
        report = Report()
        op = assignment_collection.operation_count
        total = op['adds'] + op['muls'] + op['divs']
        report.add(ReportElement("OriginalTerm", '-', op['adds'], op['muls'], op['divs'], total))
        for t in self._rules:
            start_time = timeit.default_timer()
            assignment_collection = t(assignment_collection)
            end_time = timeit.default_timer()
            op = assignment_collection.operation_count
            time_str = f"{(end_time - start_time) * 1000:.2f} ms"
            total = op['adds'] + op['muls'] + op['divs']
            report.add(ReportElement(t.__name__, time_str, op['adds'], op['muls'], op['divs'], total))
        return report

    def show_intermediate_results(self, assignment_collection: AssignmentCollection,
                                  symbols: Optional[Sequence[sp.Symbol]] = None) -> Any:
        """Shows the assignment collection after the application of each rule as HTML report for Jupyter notebook.

        Args:
            assignment_collection: the collection to apply the rules to
            symbols: if not None, only the assignments are shown that have one of these symbols as left hand side
        """
        class IntermediateResults:
            def __init__(self, strategy, collection, restrict_symbols):
                self.strategy = strategy
                self.assignment_collection = collection
                self.restrict_symbols = restrict_symbols

            def __str__(self):
                def print_assignment_collection(title, c):
                    text = title
                    if self.restrict_symbols:
                        text += "\n".join([str(e) for e in c.new_filtered(self.restrict_symbols).main_assignments])
                    else:
                        text += (" " * 3 + (" " * 3).join(str(c).splitlines(True)))
                    return text

                result = print_assignment_collection("Initial Version", self.assignment_collection)
                collection = self.assignment_collection
                for rule in self.strategy.rules:
                    collection = rule(collection)
                    result += print_assignment_collection(rule.__name__, collection)
                return result

            def _repr_html_(self):
                def print_assignment_collection(title, c):
                    text = f'<h5 style="padding-bottom:10px">{title}</h5> <div style="padding-left:20px;">'
                    if self.restrict_symbols:
                        text += "\n".join(["$$" + sp.latex(e) + '$$'
                                           for e in c.new_filtered(self.restrict_symbols).main_assignments])
                    else:
                        # noinspection PyProtectedMember
                        text += c._repr_html_()
                    text += "</div>"
                    return text

                result = print_assignment_collection("Initial Version", self.assignment_collection)
                collection = self.assignment_collection
                for rule in self.strategy.rules:
                    collection = rule(collection)
                    result += print_assignment_collection(rule.__name__, collection)
                return result

        return IntermediateResults(self, assignment_collection, symbols)

    def __repr__(self):
        result = "Simplification Strategy:\n"
        for t in self._rules:
            result += f" - {t.__name__}\n"
        return result
