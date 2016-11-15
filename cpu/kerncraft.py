from pystencils.transformations import makeLoopOverDomain, typingFromSympyInspection, \
    typeAllEquations, moveConstantsBeforeLoop, getOptimalLoopOrdering
import pystencils.ast as ast
from pystencils.backends.cbackend import CBackend, CustomSympyPrinter
from pystencils import TypedSymbol


def createKerncraftCode(listOfEquations, typeForSymbol=None, ghostLayers=None):
    """
    Creates an abstract syntax tree for a kernel function, by taking a list of update rules.

    Loops are created according to the field accesses in the equations.

    :param listOfEquations: list of sympy equations, containing accesses to :class:`pystencils.field.Field`.
           Defining the update rules of the kernel
    :param typeForSymbol: a map from symbol name to a C type specifier. If not specified all symbols are assumed to
           be of type 'double' except symbols which occur on the left hand side of equations where the
           right hand side is a sympy Boolean which are assumed to be 'bool' .
    :param ghostLayers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                        if None, the number of ghost layers is determined automatically and assumed to be equal for a
                        all dimensions

    :return: :class:`pystencils.ast.KernelFunction` node
    """
    if not typeForSymbol:
        typeForSymbol = typingFromSympyInspection(listOfEquations, "double")

    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)

    optimalLoopOrder = getOptimalLoopOrdering(allFields)
    cstyleLoopOrder = list(range(len(optimalLoopOrder)))

    body = ast.Block(assignments)
    code = makeLoopOverDomain(body, "kerncraft", ghostLayers=ghostLayers, loopOrder=cstyleLoopOrder)
    moveConstantsBeforeLoop(code)
    loopBody = code.body

    printer = CBackend(sympyPrinter=ArraySympyPrinter())

    FIXED_SIZES = ("XS", "YS", "ZS", "E1S", "E2S")

    result = ""
    for field in allFields:
        sizesPermutation = [FIXED_SIZES[i] for i in field.layout]
        suffix = "".join("[%s]" % (size,) for size in sizesPermutation)
        result += "%s%s;\n" % (field.name, suffix)

    # add parameter definitions
    for s in loopBody.undefinedSymbols:
        if isinstance(s, TypedSymbol):
            result += "%s %s;\n" % (s.dtype, s.name)

    for element in loopBody.args:
        result += printer(element)
        result += "\n"
    return result


class ArraySympyPrinter(CustomSympyPrinter):

    def _print_Access(self, fieldAccess):
        """"""
        Loop = ast.LoopOverCoordinate
        coordinateValues = [Loop.getLoopCounterSymbol(i) + offset for i, offset in enumerate(fieldAccess.offsets)]
        coordinateValues += list(fieldAccess.index)
        permutedCoordinates = [coordinateValues[i] for i in fieldAccess.field.layout]

        suffix = "".join("[%s]" % (self._print(a)) for a in permutedCoordinates)
        return "%s%s" % (self._print(fieldAccess.field.name), suffix)
