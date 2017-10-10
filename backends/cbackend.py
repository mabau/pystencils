import sympy as sp
try:
    from sympy.utilities.codegen import CCodePrinter
except ImportError:
    from sympy.printing.ccode import C99CodePrinter as CCodePrinter

from collections import namedtuple
from sympy.core.mul import _keep_coeff
from sympy.core import S

from pystencils.astnodes import Node, ResolvedFieldAccess, SympyAssignment
from pystencils.data_types import createType, PointerType, getTypeOfExpression, VectorType, castFunc
from pystencils.backends.simd_instruction_sets import selectedInstructionSet


def generateC(astNode, signatureOnly=False):
    """
    Prints the abstract syntax tree as C function
    """
    fieldTypes = set([f.dtype for f in astNode.fieldsAccessed])
    useFloatConstants = createType("double") not in fieldTypes

    vectorIS = selectedInstructionSet['double']
    printer = CBackend(constantsAsFloats=useFloatConstants, signatureOnly=signatureOnly, vectorInstructionSet=vectorIS)
    return printer(astNode)


def getHeaders(astNode):
    headers = set()

    if hasattr(astNode, 'headers'):
        headers.update(astNode.headers)
    elif isinstance(astNode, SympyAssignment):
        if type(getTypeOfExpression(astNode.rhs)) is VectorType:
            headers.update(selectedInstructionSet['double']['headers'])

    for a in astNode.args:
        if isinstance(a, Node):
            headers.update(getHeaders(a))

    return headers


# --------------------------------------- Backend Specific Nodes -------------------------------------------------------


class CustomCppCode(Node):
    def __init__(self, code, symbolsRead, symbolsDefined):
        self._code = "\n" + code
        self._symbolsRead = set(symbolsRead)
        self._symbolsDefined = set(symbolsDefined)
        self.headers = []

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
    def undefinedSymbols(self):
        return self.symbolsDefined - self._symbolsRead


class PrintNode(CustomCppCode):
    def __init__(self, symbolToPrint):
        code = '\nstd::cout << "%s  =  " << %s << std::endl; \n' % (symbolToPrint.name, symbolToPrint.name)
        super(PrintNode, self).__init__(code, symbolsRead=[symbolToPrint], symbolsDefined=set())
        self.headers.append("<iostream>")


# ------------------------------------------- Printer ------------------------------------------------------------------


class CBackend(object):

    def __init__(self, constantsAsFloats=False, sympyPrinter=None, signatureOnly=False, vectorInstructionSet=None):
        if sympyPrinter is None:
            self.sympyPrinter = CustomSympyPrinter(constantsAsFloats)
            if vectorInstructionSet is not None:
                self.sympyPrinter = VectorizedCustomSympyPrinter(vectorInstructionSet, constantsAsFloats)
            else:
                self.sympyPrinter = CustomSympyPrinter(constantsAsFloats)
        else:
            self.sympyPrinter = sympyPrinter

        self._vectorInstructionSet = vectorInstructionSet
        self._indent = "   "
        self._signatureOnly = signatureOnly

    def __call__(self, node):
        prevIs = VectorType.instructionSet
        VectorType.instructionSet = self._vectorInstructionSet
        result = str(self._print(node))
        VectorType.instructionSet = prevIs
        return result

    def _print(self, node):
        for cls in type(node).__mro__:
            methodName = "_print_" + cls.__name__
            if hasattr(self, methodName):
                return getattr(self, methodName)(node)
        raise NotImplementedError("CBackend does not support node of type " + cls.__name__)

    def _print_KernelFunction(self, node):
        functionArguments = ["%s %s" % (str(s.dtype), s.name) for s in node.parameters]
        funcDeclaration = "FUNC_PREFIX void %s(%s)" % (node.functionName, ", ".join(functionArguments))
        if self._signatureOnly:
            return funcDeclaration

        body = self._print(node.body)
        return funcDeclaration + "\n" + body

    def _print_Block(self, node):
        blockContents = "\n".join([self._print(child) for child in node.args])
        return "{\n%s\n}" % (self._indent + self._indent.join(blockContents.splitlines(True)))

    def _print_PragmaBlock(self, node):
        return "%s\n%s" % (node.pragmaLine, self._print_Block(node))

    def _print_LoopOverCoordinate(self, node):
        counterVar = node.loopCounterName
        start = "int %s = %s" % (counterVar, self.sympyPrinter.doprint(node.start))
        condition = "%s < %s" % (counterVar, self.sympyPrinter.doprint(node.stop))
        update = "%s += %s" % (counterVar, self.sympyPrinter.doprint(node.step),)
        loopStr = "for (%s; %s; %s)" % (start, condition, update)

        prefix = "\n".join(node.prefixLines)
        if prefix:
            prefix += "\n"
        return "%s%s\n%s" % (prefix, loopStr, self._print(node.body))

    def _print_SympyAssignment(self, node):
        if node.isDeclaration:
            dtype = "const " + str(node.lhs.dtype) + " " if node.isConst else str(node.lhs.dtype) + " "
            return "%s %s = %s;" % (dtype, self.sympyPrinter.doprint(node.lhs), self.sympyPrinter.doprint(node.rhs))
        else:
            lhsType = getTypeOfExpression(node.lhs)
            if type(lhsType) is VectorType and node.lhs.func == castFunc:
                return self._vectorInstructionSet['storeU'].format("&" + self.sympyPrinter.doprint(node.lhs.args[0]),
                                                                   self.sympyPrinter.doprint(node.rhs)) + ';'
            else:
                return "%s = %s;" % (self.sympyPrinter.doprint(node.lhs), self.sympyPrinter.doprint(node.rhs))

    def _print_TemporaryMemoryAllocation(self, node):
        return "%s %s = new %s[%s];" % (node.symbol.dtype, self.sympyPrinter.doprint(node.symbol.name),
                                        node.symbol.dtype.baseType, self.sympyPrinter.doprint(node.size))

    def _print_TemporaryMemoryFree(self, node):
        return "delete [] %s;" % (self.sympyPrinter.doprint(node.symbol.name),)

    def _print_CustomCppCode(self, node):
        return node.code

    def _print_Conditional(self, node):
        conditionExpr = self.sympyPrinter.doprint(node.conditionExpr)
        trueBlock = self._print_Block(node.trueBlock)
        result = "if (%s) \n %s " % (conditionExpr, trueBlock)
        if node.falseBlock:
            falseBlock = self._print_Block(node.falseBlock)
            result += "else " + falseBlock
        return result


# ------------------------------------------ Helper function & classes -------------------------------------------------


class CustomSympyPrinter(CCodePrinter):

    def __init__(self, constantsAsFloats=False):
        self._constantsAsFloats = constantsAsFloats
        super(CustomSympyPrinter, self).__init__()

    def _print_Pow(self, expr):
        """Don't use std::pow function, for small integer exponents, write as multiplication"""
        if expr.exp.is_integer and expr.exp.is_number and 0 < expr.exp < 8:
            return "(" + self._print(sp.Mul(*[expr.base] * expr.exp, evaluate=False)) + ")"
        else:
            return super(CustomSympyPrinter, self)._print_Pow(expr)

    def _print_Rational(self, expr):
        """Evaluate all rationals i.e. print 0.25 instead of 1.0/4.0"""
        res = str(expr.evalf().num)
        if self._constantsAsFloats:
            res += "f"
        return res

    def _print_Equality(self, expr):
        """Equality operator is not printable in default printer"""
        return '((' + self._print(expr.lhs) + ") == (" + self._print(expr.rhs) + '))'

    def _print_Piecewise(self, expr):
        """Print piecewise in one line (remove newlines)"""
        result = super(CustomSympyPrinter, self)._print_Piecewise(expr)
        return result.replace("\n", "")

    def _print_Float(self, expr):
        res = str(expr)
        if self._constantsAsFloats:
            res += "f"
        return res

    def _print_Function(self, expr):
        if expr.func == castFunc:
            arg, type = expr.args
            return "*((%s)(& %s))" % (PointerType(type), self._print(arg))
        else:
            return super(CustomSympyPrinter, self)._print_Function(expr)


class VectorizedCustomSympyPrinter(CustomSympyPrinter):
    SummandInfo = namedtuple("SummandInfo", ['sign', 'term'])

    def __init__(self, instructionSet, constantsAsFloats=False):
        super(VectorizedCustomSympyPrinter, self).__init__(constantsAsFloats)
        self.instructionSet = instructionSet

    def _print_Function(self, expr):
        if expr.func == castFunc:
            arg, dtype = expr.args
            if type(dtype) is VectorType:
                if type(arg) is ResolvedFieldAccess:
                    return self.instructionSet['loadU'].format("& " + self._print(arg))
                else:
                    return self.instructionSet['makeVec'].format(self._print(arg))

        return super(VectorizedCustomSympyPrinter, self)._print_Function(expr)

    def _print_Add(self, expr, order=None):
        exprType = getTypeOfExpression(expr)
        if type(exprType) is not VectorType:
            return super(VectorizedCustomSympyPrinter, self)._print_Add(expr, order)
        assert self.instructionSet['width'] == exprType.width

        summands = []
        for term in expr.args:
            if term.func == sp.Mul:
                sign, t = self._print_Mul(term, insideAdd=True)
            else:
                t = self._print(term)
                sign = 1
            summands.append(self.SummandInfo(sign, t))
        # Use positive terms first
        summands.sort(key=lambda e: e.sign, reverse=True)
        # if no positive term exists, prepend a zero
        if summands[0].sign == -1:
            summands.insert(0, self.SummandInfo(1, "0"))

        assert len(summands) >= 2
        processed = summands[0].term
        for summand in summands[1:]:
            func = self.instructionSet['-'] if summand.sign == -1 else self.instructionSet['+']
            processed = func.format(processed, summand.term)
        return processed

    def _print_Mul(self, expr, insideAdd=False):
        exprType = getTypeOfExpression(expr)
        if type(exprType) is not VectorType:
            return super(VectorizedCustomSympyPrinter, self)._print_Mul(expr)
        assert self.instructionSet['width'] == exprType.width

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = -1
        else:
            sign = 1

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        # Gather args for numerator/denominator
        for item in expr.as_ordered_factors():
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(sp.Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(sp.Pow(item.base, -item.exp))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self._print(x) for x in a]
        b_str = [self._print(x) for x in b]

        result = a_str[0]
        for item in a_str[1:]:
            result = self.instructionSet['*'].format(result, item)

        if len(b) > 0:
            denominator_str = b_str[0]
            for item in b_str[1:]:
                denominator_str = self.instructionSet['*'].format(denominator_str, item)
            result = self.instructionSet['/'].format(result, denominator_str)

        if insideAdd:
            return sign, result
        else:
            if sign < 0:
                return self.instructionSet['*'].format(self._print(S.NegativeOne), result)
            else:
                return result

    def _print_Relational(self, expr):
        exprType = getTypeOfExpression(expr)
        if type(exprType) is not VectorType:
            return super(VectorizedCustomSympyPrinter, self)._print_Relational(expr)
        assert self.instructionSet['width'] == exprType.width

        return self.instructionSet[expr.rel_op].format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_Equality(self, expr):
        exprType = getTypeOfExpression(expr)
        if type(exprType) is not VectorType:
            return super(VectorizedCustomSympyPrinter, self)._print_Equality(expr)
        assert self.instructionSet['width'] == exprType.width

        return self.instructionSet['=='].format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_Piecewise(self, expr):
        exprType = getTypeOfExpression(expr)
        if type(exprType) is not VectorType:
            return super(VectorizedCustomSympyPrinter, self)._print_Piecewise(expr)
        assert self.instructionSet['width'] == exprType.width

        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")

        result = self._print(expr.args[-1][0])
        for trueExpr, condition in reversed(expr.args[:-1]):
            result = self.instructionSet['blendv'].format(result, self._print(trueExpr), self._print(condition))
        return result



