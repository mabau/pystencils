import sympy as sp
import textwrap
from collections import namedtuple


class SimplificationStrategy:
    """
    A simplification strategy is an ordered collection of simplification rules.
    Each simplification is a function taking an equation collection, and returning a new simplified
    equation collection. The strategy can nicely print intermediate simplification stages and results
    to Jupyter notebooks.
    """

    def __init__(self):
        self._rules = []

    def addSimplificationRule(self, rule):
        """
        Adds the given simplification rule to the end of the collection.
        :param rule: function that taking one equation collection and returning a (simplified) equation collection
        """
        self._rules.append(rule)

    @property
    def rules(self):
        return self._rules

    def apply(self, updateRule):
        """Applies all simplification rules to the given equation collection"""
        for t in self._rules:
            updateRule = t(updateRule)
        return updateRule

    def __call__(self, equationCollection):
        """Same as apply"""
        return self.apply(equationCollection)

    def createSimplificationReport(self, equationCollection):
        """
        Returns a simplification report containing the number of operations at each simplification stage, together
        with the run-time the simplification took.
        """

        ReportElement = namedtuple('ReportElement', ['simplificationName', 'adds', 'muls', 'divs', 'runtime'])

        class Report:
            def __init__(self):
                self.elements = []

            def add(self, element):
                self.elements.append(element)

            def __str__(self):
                try:
                    import tabulate
                    return tabulate(self.elements, headers=['Name', 'Adds', 'Muls', 'Divs', 'Runtime'])
                except ImportError:
                    result = "Name, Adds, Muls, Divs, Runtime\n"
                    for e in self.elements:
                        result += ",".join(e) + "\n"
                    return result

            def _repr_html_(self):
                htmlTable = '<table style="border:none">'
                htmlTable += "<tr> <th>Name</th> <th>Adds</th> <th>Muls</th> <th>Divs</th> <th>Runtime</th></tr>"
                line = "<tr><td>{simplificationName}</td>" \
                       "<td>{adds}</td> <td>{muls}</td> <td>{divs}</td>  <td>{runtime}</td> </tr>"

                for e in self.elements:
                    htmlTable += line.format(**e._asdict())
                htmlTable += "</table>"
                return htmlTable

        import time
        report = Report()
        op = equationCollection.operationCount
        report.add(ReportElement("OriginalTerm", op['adds'], op['muls'], op['divs'], '-'))
        for t in self._rules:
            startTime = time.perf_counter()
            equationCollection = t(equationCollection)
            endTime = time.perf_counter()
            op = equationCollection.operationCount
            timeStr = "%.2f ms" % ((endTime - startTime) * 1000,)
            report.add(ReportElement(t.__name__, op['adds'], op['muls'], op['divs'], timeStr))
        return report

    def showIntermediateResults(self, equationCollection, symbols=None):

        class IntermediateResults:
            def __init__(self, strategy, eqColl, resSyms):
                self.strategy = strategy
                self.equationCollection = eqColl
                self.restrictSymbols = resSyms

            def __str__(self):
                def printEqCollection(title, eqColl):
                    text = title
                    if self.restrictSymbols:
                        text += "\n".join([str(e) for e in eqColl.get(self.restrictSymbols)])
                    else:
                        text += textwrap.indent(str(eqColl), " " * 3)
                    return text

                result = printEqCollection("Initial Version", self.equationCollection)
                eqColl = self.equationCollection
                for rule in self.strategy.rules:
                    eqColl = rule(eqColl)
                    result += printEqCollection(rule.__name__, eqColl)
                return result

            def _repr_html_(self):
                def printEqCollection(title, eqColl):
                    text = '<h5 style="padding-bottom:10px">%s</h5> <div style="padding-left:20px;">' % (title, )
                    if self.restrictSymbols:
                        text += "\n".join(["$$" + sp.latex(e) + '$$' for e in eqColl.get(self.restrictSymbols)])
                    else:
                        text += eqColl._repr_html_()
                    text += "</div>"
                    return text

                result = printEqCollection("Initial Version", self.equationCollection)
                eqColl = self.equationCollection
                for rule in self.strategy.rules:
                    eqColl = rule(eqColl)
                    result += printEqCollection(rule.__name__, eqColl)
                return result

        return IntermediateResults(self, equationCollection, symbols)

    def __repr__(self):
        result = "Simplification Strategy:\n"
        for t in self._rules:
            result += " - %s\n" % (t.__name__,)
        return result
