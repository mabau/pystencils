from collections import namedtuple

import sympy as sp
from sympy.core import S
from sympy.utilities.codegen import CCodePrinter
from sympy.core.mul import _keep_coeff

from pystencils.backends.cbackend import CustomSympyPrinter
from pystencils.types import getBaseType, createTypeFromString


def getInstructionSetInfoIntel(dataType='double', instructionSet='avx'):
    baseNames = {
        '+': 'add[0, 1]',
        '-': 'sub[0, 1]',
        '*': 'mul[0, 1]',
        '/': 'div[0, 1]',

        '==': 'cmp[0, 1, _CMP_EQ_UQ  ]',
        '!=': 'cmp[0, 1, _CMP_NEQ_UQ ]',
        '>=': 'cmp[0, 1, _CMP_GE_OQ  ]',
        '<=': 'cmp[0, 1, _CMP_LE_OQ  ]',
        '<': 'cmp[0, 1, _CMP_NGE_UQ ]',
        '>': 'cmp[0, 1, _CMP_NLE_UQ ]',

        'blendv': 'blendv[0, 1, 2]',

        'sqrt': 'sqrt[0]',

        'makeVec':  'set[0,0,0,0]',
        'makeZero': 'setzero[]',

        'loadU': 'loadu [0]',
        'loadA': 'load [0]',
        'storeU': 'storeu[0]',
        'storeA': 'store [0]',
    }

    suffix = {
        'double': 'pd',
        'float': 'ps',
    }
    prefix = {
        'sse': '_mm',
        'avx': '_mm256',
        'avx512': '_mm512',
    }

    width = {
        ("double", "sse"): 2,
        ("float", "sse"): 4,
        ("double", "avx"): 4,
        ("float", "avx"): 8,
        ("double", "avx512"): 8,
        ("float", "avx512"): 16,
    }

    result = {}
    pre = prefix[instructionSet]
    suf = suffix[dataType]
    for intrinsicId, functionShortcut in baseNames.items():
        functionShortcut = functionShortcut.strip()
        name = functionShortcut[:functionShortcut.index('[')]
        args = functionShortcut[functionShortcut.index('[') + 1: -1]
        argString = "("
        for arg in args.split(","):
            arg = arg.strip()
            if not arg:
                continue
            if arg in ('0', '1', '2', '3', '4', '5'):
                argString += "{" + arg + "},"
            else:
                argString += arg
        argString = argString[:-1] + ")"
        result[intrinsicId] = pre + "_" + name + "_" + suf + argString

    result['width'] = width[(dataType, instructionSet)]
    result['dataTypePrefix'] = {
        'double': "_" + pre + 'd',
        'float': "_" + pre,
    }

    return result


class VectorizedCBackend(object):

    def __init__(self, astNode, instructionSet='avx'):
        fieldTypes = set([getBaseType(f.dtype) for f in astNode.fieldsAccessed])
        if len(fieldTypes) != 1:
            raise ValueError("Vectorized backend does not support kernels with mixed field types")
        fieldType = fieldTypes.pop()
        assert fieldType.is_float
        dtypeName = str(fieldType)

        instructionSetInfo = getInstructionSetInfoIntel(dtypeName, instructionSet)

        self.vectorizationWidth = instructionSetInfo['width']
        self.sympyVecPrinter = CustomSympyPrinterVectorized(instructionSetInfo)
        self.sympyPrinter = CustomSympyPrinter(constantsAsFloats=(dtypeName == 'float'))

        self._indent = "   "
        self._vecTypeName = instructionSetInfo['dataTypePrefix'][dtypeName]
        self.dtypeName = dtypeName

    def __call__(self, node):
        return str(self._print(node))

    def _print(self, node):
        for cls in type(node).__mro__:
            methodName = "_print_" + cls.__name__
            if hasattr(self, methodName):
                return getattr(self, methodName)(node)
        raise NotImplementedError("CBackend does not support node of type " + cls.__name__)

    def _print_KernelFunction(self, node):
        blockContents = "\n".join([self._print(child) for child in node.body.args])
        constantBlock = self.sympyVecPrinter.getConstantsBlock(self._vecTypeName)

        body = "{\n%s\n%s\n}" % (constantBlock, self._indent + self._indent.join(blockContents.splitlines(True)))

        functionArguments = ["%s %s" % (str(s.dtype), s.name) for s in node.parameters]
        funcDeclaration = "FUNC_PREFIX void %s(%s)" % (node.functionName, ", ".join(functionArguments))
        return funcDeclaration + "\n" + body

    def _print_Block(self, node):
        blockContents = "\n".join([self._print(child) for child in node.args])
        return "{\n%s\n}" % (self._indent + self._indent.join(blockContents.splitlines(True)),)

    def _print_PragmaBlock(self, node):
        return "%s\n%s" % (node.pragmaLine, self._print_Block(node))

    def _print_LoopOverCoordinate(self, node):
        if node.isInnermostLoop:
            iterRange = node.stop - node.start
            if isinstance(iterRange, sp.Basic) and not iterRange.is_integer:
                raise NotImplementedError("Vectorized backend currently only supports fixed size inner loops")
            if iterRange % self.vectorizationWidth != 0 or node.step != 1:
                raise NotImplementedError("Vectorized backend only supports loop bounds that are "
                                          "multiples of vectorization width")
            step = self.vectorizationWidth
        else:
            step = node.step

        counterVar = node.loopCounterName
        start = "int %s = %s" % (counterVar, self.sympyPrinter.doprint(node.start))
        condition = "%s < %s" % (counterVar, self.sympyPrinter.doprint(node.stop))
        update = "%s += %s" % (counterVar, self.sympyPrinter.doprint(step),)
        loopStr = "for (%s; %s; %s)" % (start, condition, update)

        prefix = "\n".join(node.prefixLines)
        if prefix:
            prefix += "\n"
        return "%s%s\n%s" % (prefix, loopStr, self._print(node.body))

    def _print_SympyAssignment(self, node):
        dtype = ""
        if node.isDeclaration:
            assert str(getBaseType(node.lhs.dtype)) in (self.dtypeName, 'bool')
            if node.lhs.dtype == createTypeFromString(self.dtypeName):
                dtypeStr = self._vecTypeName
                printer = self.sympyVecPrinter
            else:
                dtypeStr = str(node.lhs.dtype)
                printer = self.sympyPrinter

            if node.isConst:
                dtype = "const " + dtypeStr + " "
            else:
                dtype = dtypeStr + " "
        else:
            printer = self.sympyVecPrinter
        return "%s %s = %s;" % (str(dtype), printer.doprint(node.lhs), printer.doprint(node.rhs))

    def _print_TemporaryMemoryAllocation(self, node):
        return "%s * %s = new %s[%s];" % (node.symbol.dtype, self.sympyPrinter.doprint(node.symbol),
                                         node.symbol.dtype, self.sympyPrinter.doprint(node.size))

    def _print_TemporaryMemoryFree(self, node):
        return "delete [] %s;" % (self.sympyPrinter.doprint(node.symbol),)

    def _print_CustomCppCode(self, node):
        return node.code


class CustomSympyPrinterVectorized(CCodePrinter):
    SummandInfo = namedtuple("SummandInfo", ['sign', 'term'])

    def __init__(self, instructionSetInfo):
        super(CustomSympyPrinterVectorized, self).__init__()
        self.intrinsics = instructionSetInfo
        self.constantsDict = {}

    def getConstantsBlock(self, vecTypeStr):
        result = ""
        for value, symbol in self.constantsDict.items():
            rhsStr = self.intrinsics['makeVec'].format(self._print(value))
            result += "const %s %s = %s;\n" % (vecTypeStr, symbol.name, rhsStr)
        return result

    def _print_Add(self, expr, order=None):
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
            func = self.intrinsics['-'] if summand.sign == -1 else self.intrinsics['+']
            processed = func.format(processed, summand.term)
        return processed

    def _print_Mul(self, expr, insideAdd=False):

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
            result = self.intrinsics['*'].format(result, item)

        if len(b) > 0:
            denominator_str = b_str[0]
            for item in b_str[1:]:
                denominator_str = self.intrinsics['*'].format(denominator_str, item)
            result = self.intrinsics['/'].format(result, denominator_str)

        if insideAdd:
            return sign, result
        else:
            if sign < 0:
                return self.intrinsics['*'].format(self._print(S.NegativeOne), result)
            else:
                return result

    def _print_Pow(self, expr):
        """Don't use std::pow function, for small integer exponents, write as multiplication"""
        if expr.exp.is_integer and expr.exp.is_number and 0 < expr.exp < 8:
            return self._print(sp.Mul(*[expr.base] * expr.exp, evaluate=False))
        else:
            return super(CustomSympyPrinterVectorized, self)._print_Pow(expr)

    def _print_Float(self, expr):
        if expr not in self.constantsDict:
            self.constantsDict[expr] = sp.Dummy()
        symbol = self.constantsDict[expr]
        return symbol.name

    def _print_Rational(self, expr):
        if expr not in self.constantsDict:
            self.constantsDict[expr] = sp.Symbol("__value_%d_%d" % (expr.p, expr.q))
        symbol = self.constantsDict[expr]
        return symbol.name

    def _print_Piecewise(self, expr):
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
            result = self.intrinsics['blendv'].format(result, self._print(trueExpr), self._print(condition))
        return result

    def _print_Relational(self, expr):
        return self.intrinsics[expr.rel_op].format(expr.lhs, expr.rhs)

    def _print_Equality(self, expr):
        """Equality operator is not printable in default printer"""
        return self.intrinsics['=='].format(self._print(expr.lhs), self._print(expr.rhs))
