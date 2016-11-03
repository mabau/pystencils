import cgen as c
from sympy.utilities.codegen import CCodePrinter
from pystencils.ast import Node


def printCCode(astNode):
    """
    Prints the abstract syntax tree as C function
    """
    printer = CBackend(cuda=False)
    return printer(astNode)


def printCudaCode(astNode):
    printer = CBackend(cuda=True)
    return printer(astNode)

# --------------------------------------- Backend Specific Nodes -------------------------------------------------------


class CustomCppCode(Node):
    def __init__(self, code, symbolsRead, symbolsDefined):
        self._code = "\n" + code
        self._symbolsRead = set(symbolsRead)
        self._symbolsDefined = set(symbolsDefined)

    @property
    def code(self):
        return self._code

    @property
    def args(self):
        return []

    @property
    def symbolsDefined(self):
        return self._symbolsDefined

    @property
    def symbolsRead(self):
        return self._symbolsRead

    def generateC(self):
        return c.LiteralLines(self._code)


class PrintNode(CustomCppCode):
    def __init__(self, symbolToPrint):
        code = '\nstd::cout << "%s  =  " << %s << std::endl; \n' % (symbolToPrint.name, symbolToPrint.name)
        super(PrintNode, self).__init__(code, symbolsRead=[symbolToPrint], symbolsDefined=set())


# ------------------------------------------- Printer ------------------------------------------------------------------


class CBackend:

    def __init__(self, cuda=False):
        self.cuda = cuda
        self.sympyPrinter = CustomSympyPrinter()

    def __call__(self, node):
        return self._print(node)

    def _print(self, node):
        for cls in type(node).__mro__:
            methodName = "_print_" + cls.__name__
            if hasattr(self, methodName):
                return getattr(self, methodName)(node)
        raise NotImplementedError("CBackend does not support node of type " + cls.__name__)

    def _print_KernelFunction(self, node):
        functionArguments = [MyPOD(s.dtype, s.name) for s in node.parameters]
        prefix = "__global__ void" if self.cuda else "void"
        functionPOD = MyPOD(prefix, node.functionName, )
        funcDeclaration = c.FunctionDeclaration(functionPOD, functionArguments)
        return c.FunctionBody(funcDeclaration, self._print(node.body))

    def _print_Block(self, node):
        return c.Block([self._print(child) for child in node.args])

    def _print_PragmaBlock(self, node):
        class PragmaGenerable(c.Generable):
            def __init__(self, line, block):
                self._line = line
                self._block = block

            def generate(self):
                yield self._line
                for e in self._block.generate():
                    yield e
        return PragmaGenerable(node.pragmaLine, self._print_Block(node))

    def _print_LoopOverCoordinate(self, node):
        class LoopWithOptionalPrefix(c.CustomLoop):
            def __init__(self, intro_line, body, prefixLines=[]):
                super(LoopWithOptionalPrefix, self).__init__(intro_line, body)
                self.prefixLines = prefixLines

            def generate(self):
                for l in self.prefixLines:
                    yield l

                for e in super(LoopWithOptionalPrefix, self).generate():
                    yield e

        counterVar = node.loopCounterName
        start = "int %s = %d" % (counterVar, node.ghostLayers)
        condition = "%s < %s" % (counterVar, self.sympyPrinter.doprint(node.iterationEnd))
        update = "++%s" % (counterVar,)
        loopStr = "for (%s; %s; %s)" % (start, condition, update)
        return LoopWithOptionalPrefix(loopStr, self._print(node.body), prefixLines=node.prefixLines)

    def _print_SympyAssignment(self, node):
        dtype = ""
        if node.isDeclaration:
            if node.isConst:
                dtype = "const " + node.lhs.dtype + " "
            else:
                dtype = node.lhs.dtype + " "

        return c.Assign(dtype + self.sympyPrinter.doprint(node.lhs),
                        self.sympyPrinter.doprint(node.rhs))

    def _print_TemporaryMemoryAllocation(self, node):
        return c.Assign(node.symbol.dtype + " * " + self.sympyPrinter.doprint(node.symbol),
                        "new %s[%s]" % (node.symbol.dtype, self.sympyPrinter.doprint(node.size)))

    def _print_TemporaryMemoryFree(self, node):
        return c.Statement("delete [] %s" % (self.sympyPrinter.doprint(node.symbol),))

    def _print_CustomCppCode(self, node):
        return c.LiteralLines(node.code)


# ------------------------------------------ Helper function & classes -------------------------------------------------


class CustomSympyPrinter(CCodePrinter):
    def _print_Pow(self, expr):
        """Don't use std::pow function, for small integer exponents, write as multiplication"""
        if expr.exp.is_integer and expr.exp.is_number and 0 < expr.exp < 8:
            return '(' + '*'.join(["(" + self._print(expr.base) + ")"] * expr.exp) + ')'
        else:
            return super(CustomSympyPrinter, self)._print_Pow(expr)

    def _print_Rational(self, expr):
        """Evaluate all rationals i.e. print 0.25 instead of 1.0/4.0"""
        return str(expr.evalf().num)

    def _print_Equality(self, expr):
        """Equality operator is not printable in default printer"""
        return '((' + self._print(expr.lhs) + ") == (" + self._print(expr.rhs) + '))'

    def _print_Piecewise(self, expr):
        """Print piecewise in one line (remove newlines)"""
        result = super(CustomSympyPrinter, self)._print_Piecewise(expr)
        return result.replace("\n", "")


class MyPOD(c.Declarator):
    def __init__(self, dtype, name):
        self.dtype = dtype
        self.name = name

    def get_decl_pair(self):
        return [self.dtype], self.name
