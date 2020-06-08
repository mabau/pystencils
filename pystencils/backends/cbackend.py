import re
from collections import namedtuple
from typing import Set

import numpy as np
import sympy as sp
from sympy.core import S
from sympy.logic.boolalg import BooleanFalse, BooleanTrue
from sympy.printing.ccode import C89CodePrinter

from pystencils.astnodes import KernelFunction, Node
from pystencils.cpu.vectorization import vec_all, vec_any
from pystencils.data_types import (
    PointerType, VectorType, address_of, cast_func, create_type, get_type_of_expression,
    reinterpret_cast_func, vector_memory_access)
from pystencils.fast_approximation import fast_division, fast_inv_sqrt, fast_sqrt
from pystencils.integer_functions import (
    bit_shift_left, bit_shift_right, bitwise_and, bitwise_or, bitwise_xor,
    int_div, int_power_of_2, modulo_ceil)

try:
    from sympy.printing.ccode import C99CodePrinter as CCodePrinter
except ImportError:
    from sympy.printing.ccode import CCodePrinter  # for sympy versions < 1.1

__all__ = ['generate_c', 'CustomCodeNode', 'PrintNode', 'get_headers', 'CustomSympyPrinter']


HEADER_REGEX = re.compile(r'^[<"].*[">]$')

KERNCRAFT_NO_TERNARY_MODE = False


def generate_c(ast_node: Node,
               signature_only: bool = False,
               dialect='c',
               custom_backend=None,
               with_globals=True) -> str:
    """Prints an abstract syntax tree node as C or CUDA code.

    This function does not need to distinguish for most AST nodes between C, C++ or CUDA code, it just prints 'C-like'
    code as encoded in the abstract syntax tree (AST). The AST is built differently for C or CUDA by calling different
    create_kernel functions.

    Args:
        ast_node:
        signature_only:
        dialect: 'c' or 'cuda'
    Returns:
        C-like code for the ast node and its descendants
    """
    global_declarations = get_global_declarations(ast_node)
    for d in global_declarations:
        if hasattr(ast_node, "global_variables"):
            ast_node.global_variables.update(d.symbols_defined)
        else:
            ast_node.global_variables = d.symbols_defined
    if custom_backend:
        printer = custom_backend
    elif dialect == 'c':
        try:
            instruction_set = ast_node.instruction_set
        except Exception:
            instruction_set = None
        printer = CBackend(signature_only=signature_only,
                           vector_instruction_set=instruction_set)
    elif dialect == 'cuda':
        from pystencils.backends.cuda_backend import CudaBackend
        printer = CudaBackend(signature_only=signature_only)
    elif dialect == 'opencl':
        from pystencils.backends.opencl_backend import OpenClBackend
        printer = OpenClBackend(signature_only=signature_only)
    else:
        raise ValueError("Unknown dialect: " + str(dialect))
    code = printer(ast_node)
    if not signature_only and isinstance(ast_node, KernelFunction):
        if with_globals and global_declarations:
            code = "\n" + code
            for declaration in global_declarations:
                code = printer(declaration) + "\n" + code

    return code


def get_global_declarations(ast):
    global_declarations = []

    def visit_node(sub_ast):
        nonlocal global_declarations
        if hasattr(sub_ast, "required_global_declarations"):
            global_declarations += sub_ast.required_global_declarations

        if hasattr(sub_ast, "args"):
            for node in sub_ast.args:
                visit_node(node)

    visit_node(ast)

    return sorted(set(global_declarations), key=str)


def get_headers(ast_node: Node) -> Set[str]:
    """Return a set of header files, necessary to compile the printed C-like code."""
    headers = set()

    if isinstance(ast_node, KernelFunction) and ast_node.instruction_set:
        headers.update(ast_node.instruction_set['headers'])

    if hasattr(ast_node, 'headers'):
        headers.update(ast_node.headers)
    for a in ast_node.args:
        if isinstance(a, (sp.Expr, Node)):
            headers.update(get_headers(a))

    for g in get_global_declarations(ast_node):
        if isinstance(g, Node):
            headers.update(get_headers(g))

    for h in headers:
        assert HEADER_REGEX.match(h), f'header /{h}/ does not follow the pattern /"..."/ or /<...>/'

    return sorted(headers)


# --------------------------------------- Backend Specific Nodes -------------------------------------------------------


class CustomCodeNode(Node):
    def __init__(self, code, symbols_read, symbols_defined, parent=None):
        super(CustomCodeNode, self).__init__(parent=parent)
        self._code = "\n" + code
        self._symbols_read = set(symbols_read)
        self._symbols_defined = set(symbols_defined)
        self.headers = []

    def get_code(self, dialect, vector_instruction_set):
        return self._code

    @property
    def args(self):
        return []

    @property
    def symbols_defined(self):
        return self._symbols_defined

    @property
    def undefined_symbols(self):
        return self._symbols_read - self._symbols_defined

    def __eq___(self, other):
        return self._code == other._code

    def __hash__(self):
        return hash(self._code)


class PrintNode(CustomCodeNode):
    # noinspection SpellCheckingInspection
    def __init__(self, symbol_to_print):
        code = '\nstd::cout << "%s  =  " << %s << std::endl; \n' % (symbol_to_print.name, symbol_to_print.name)
        super(PrintNode, self).__init__(code, symbols_read=[symbol_to_print], symbols_defined=set())
        self.headers.append("<iostream>")


# ------------------------------------------- Printer ------------------------------------------------------------------


# noinspection PyPep8Naming
class CBackend:

    def __init__(self, sympy_printer=None, signature_only=False, vector_instruction_set=None, dialect='c'):
        if sympy_printer is None:
            if vector_instruction_set is not None:
                self.sympy_printer = VectorizedCustomSympyPrinter(vector_instruction_set)
            else:
                self.sympy_printer = CustomSympyPrinter()
        else:
            self.sympy_printer = sympy_printer

        self._vector_instruction_set = vector_instruction_set
        self._indent = "   "
        self._dialect = dialect
        self._signatureOnly = signature_only

    def __call__(self, node):
        prev_is = VectorType.instruction_set
        VectorType.instruction_set = self._vector_instruction_set
        result = str(self._print(node))
        VectorType.instruction_set = prev_is
        return result

    def _print(self, node):
        if isinstance(node, str):
            return node
        for cls in type(node).__mro__:
            method_name = "_print_" + cls.__name__
            if hasattr(self, method_name):
                return getattr(self, method_name)(node)
        raise NotImplementedError(self.__class__.__name__ + " does not support node of type " + node.__class__.__name__)

    def _print_Type(self, node):
        return str(node)

    def _print_KernelFunction(self, node):
        function_arguments = ["%s %s" % (self._print(s.symbol.dtype), s.symbol.name) for s in node.get_parameters()]
        launch_bounds = ""
        if self._dialect == 'cuda':
            max_threads = node.indexing.max_threads_per_block()
            if max_threads:
                launch_bounds = "__launch_bounds__({}) ".format(max_threads)
        func_declaration = "FUNC_PREFIX %svoid %s(%s)" % (launch_bounds, node.function_name,
                                                          ", ".join(function_arguments))
        if self._signatureOnly:
            return func_declaration

        body = self._print(node.body)
        return func_declaration + "\n" + body

    def _print_Block(self, node):
        block_contents = "\n".join([self._print(child) for child in node.args])
        return "{\n%s\n}" % (self._indent + self._indent.join(block_contents.splitlines(True)))

    def _print_PragmaBlock(self, node):
        return "%s\n%s" % (node.pragma_line, self._print_Block(node))

    def _print_LoopOverCoordinate(self, node):
        counter_symbol = node.loop_counter_name
        start = "int %s = %s" % (counter_symbol, self.sympy_printer.doprint(node.start))
        condition = "%s < %s" % (counter_symbol, self.sympy_printer.doprint(node.stop))
        update = "%s += %s" % (counter_symbol, self.sympy_printer.doprint(node.step),)
        loop_str = "for (%s; %s; %s)" % (start, condition, update)

        prefix = "\n".join(node.prefix_lines)
        if prefix:
            prefix += "\n"
        return "%s%s\n%s" % (prefix, loop_str, self._print(node.body))

    def _print_SympyAssignment(self, node):
        if node.is_declaration:
            if node.use_auto:
                data_type = 'auto '
            else:
                if node.is_const:
                    prefix = 'const '
                else:
                    prefix = ''
                data_type = prefix + self._print(node.lhs.dtype).replace(' const', '') + " "

            return "%s%s = %s;" % (data_type,
                                   self.sympy_printer.doprint(node.lhs),
                                   self.sympy_printer.doprint(node.rhs))
        else:
            lhs_type = get_type_of_expression(node.lhs)
            printed_mask = ""
            if type(lhs_type) is VectorType and isinstance(node.lhs, cast_func):
                arg, data_type, aligned, nontemporal, mask = node.lhs.args
                instr = 'storeU'
                if aligned:
                    instr = 'stream' if nontemporal else 'storeA'
                if mask != True:  # NOQA
                    instr = 'maskStore' if aligned else 'maskStoreU'
                    printed_mask = self.sympy_printer.doprint(mask)
                    if self._vector_instruction_set['dataTypePrefix']['double'] == '__mm256d':
                        printed_mask = "_mm256_castpd_si256({})".format(printed_mask)

                rhs_type = get_type_of_expression(node.rhs)
                if type(rhs_type) is not VectorType:
                    rhs = cast_func(node.rhs, VectorType(rhs_type))
                else:
                    rhs = node.rhs

                return self._vector_instruction_set[instr].format("&" + self.sympy_printer.doprint(node.lhs.args[0]),
                                                                  self.sympy_printer.doprint(rhs),
                                                                  printed_mask) + ';'
            else:
                return "%s = %s;" % (self.sympy_printer.doprint(node.lhs), self.sympy_printer.doprint(node.rhs))

    def _print_TemporaryMemoryAllocation(self, node):
        align = 64
        np_dtype = node.symbol.dtype.base_type.numpy_dtype
        required_size = np_dtype.itemsize * node.size + align
        size = modulo_ceil(required_size, align)
        code = "{dtype} {name}=({dtype})aligned_alloc({align}, {size}) + {offset};"
        return code.format(dtype=node.symbol.dtype,
                           name=self.sympy_printer.doprint(node.symbol.name),
                           size=self.sympy_printer.doprint(size),
                           offset=int(node.offset(align)),
                           align=align)

    def _print_TemporaryMemoryFree(self, node):
        align = 64
        return "free(%s - %d);" % (self.sympy_printer.doprint(node.symbol.name), node.offset(align))

    def _print_SkipIteration(self, _):
        return "continue;"

    def _print_CustomCodeNode(self, node):
        return node.get_code(self._dialect, self._vector_instruction_set)

    def _print_SourceCodeComment(self, node):
        return "/* " + node.text + " */"

    def _print_EmptyLine(self, node):
        return ""

    def _print_Conditional(self, node):
        if type(node.condition_expr) is BooleanTrue:
            return self._print_Block(node.true_block)
        elif type(node.condition_expr) is BooleanFalse:
            return self._print_Block(node.false_block)
        cond_type = get_type_of_expression(node.condition_expr)
        if isinstance(cond_type, VectorType):
            raise ValueError("Problem with Conditional inside vectorized loop - use vec_any or vec_all")
        condition_expr = self.sympy_printer.doprint(node.condition_expr)
        true_block = self._print_Block(node.true_block)
        result = "if (%s)\n%s " % (condition_expr, true_block)
        if node.false_block:
            false_block = self._print_Block(node.false_block)
            result += "else " + false_block
        return result


# ------------------------------------------ Helper function & classes -------------------------------------------------


# noinspection PyPep8Naming
class CustomSympyPrinter(CCodePrinter):

    def __init__(self):
        super(CustomSympyPrinter, self).__init__()
        self._float_type = create_type("float32")
        if 'Min' in self.known_functions:
            del self.known_functions['Min']
        if 'Max' in self.known_functions:
            del self.known_functions['Max']

    def _print_Pow(self, expr):
        """Don't use std::pow function, for small integer exponents, write as multiplication"""
        if not expr.free_symbols:
            return self._typed_number(expr.evalf(), get_type_of_expression(expr))

        if expr.exp.is_integer and expr.exp.is_number and 0 < expr.exp < 8:
            return "(" + self._print(sp.Mul(*[expr.base] * expr.exp, evaluate=False)) + ")"
        elif expr.exp.is_integer and expr.exp.is_number and - 8 < expr.exp < 0:
            return "1 / ({})".format(self._print(sp.Mul(*[expr.base] * (-expr.exp), evaluate=False)))
        else:
            return super(CustomSympyPrinter, self)._print_Pow(expr)

    def _print_Rational(self, expr):
        """Evaluate all rationals i.e. print 0.25 instead of 1.0/4.0"""
        res = str(expr.evalf().num)
        return res

    def _print_Equality(self, expr):
        """Equality operator is not printable in default printer"""
        return '((' + self._print(expr.lhs) + ") == (" + self._print(expr.rhs) + '))'

    def _print_Piecewise(self, expr):
        """Print piecewise in one line (remove newlines)"""
        result = super(CustomSympyPrinter, self)._print_Piecewise(expr)
        return result.replace("\n", "")

    def _print_Abs(self, expr):
        if expr.is_integer:
            return 'abs({0})'.format(self._print(expr.args[0]))
        else:
            return 'fabs({0})'.format(self._print(expr.args[0]))

    def _print_Type(self, node):
        return str(node)

    def _print_Function(self, expr):
        infix_functions = {
            bitwise_xor: '^',
            bit_shift_right: '>>',
            bit_shift_left: '<<',
            bitwise_or: '|',
            bitwise_and: '&',
        }
        if hasattr(expr, 'to_c'):
            return expr.to_c(self._print)
        if isinstance(expr, reinterpret_cast_func):
            arg, data_type = expr.args
            return "*((%s)(& %s))" % (self._print(PointerType(data_type, restrict=False)), self._print(arg))
        elif isinstance(expr, address_of):
            assert len(expr.args) == 1, "address_of must only have one argument"
            return "&(%s)" % self._print(expr.args[0])
        elif isinstance(expr, cast_func):
            arg, data_type = expr.args
            if isinstance(arg, sp.Number) and arg.is_finite:
                return self._typed_number(arg, data_type)
            else:
                if str(arg) == "-1":
                    print("!!")
                return "((%s)(%s))" % (data_type, self._print(arg))
        elif isinstance(expr, fast_division):
            return "({})".format(self._print(expr.args[0] / expr.args[1]))
        elif isinstance(expr, fast_sqrt):
            return "({})".format(self._print(sp.sqrt(expr.args[0])))
        elif isinstance(expr, vec_any) or isinstance(expr, vec_all):
            return self._print(expr.args[0])
        elif isinstance(expr, fast_inv_sqrt):
            return "({})".format(self._print(1 / sp.sqrt(expr.args[0])))
        elif isinstance(expr, sp.Abs):
            return "abs({})".format(self._print(expr.args[0]))
        elif isinstance(expr, sp.Mod):
            if expr.args[0].is_integer and expr.args[1].is_integer:
                return "({} % {})".format(self._print(expr.args[0]), self._print(expr.args[1]))
            else:
                return "fmod({}, {})".format(self._print(expr.args[0]), self._print(expr.args[1]))
        elif expr.func in infix_functions:
            return "(%s %s %s)" % (self._print(expr.args[0]), infix_functions[expr.func], self._print(expr.args[1]))
        elif expr.func == int_power_of_2:
            return "(1 << (%s))" % (self._print(expr.args[0]))
        elif expr.func == int_div:
            return "((%s) / (%s))" % (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            name = expr.name if hasattr(expr, 'name') else expr.__class__.__name__
            arg_str = ', '.join(self._print(a) for a in expr.args)
            return f'{name}({arg_str})'

    def _typed_number(self, number, dtype):
        res = self._print(number)
        if dtype.numpy_dtype == np.float32:
            return res + '.0f' if '.' not in res else res + 'f'
        elif dtype.numpy_dtype == np.float64:
            return res + '.0' if '.' not in res else res
        else:
            return res

    def _print_Sum(self, expr):
        template = """[&]() {{
    {dtype} sum = ({dtype}) 0;
    for ( {iterator_dtype} {var} = {start}; {condition}; {var} += {increment} ) {{
        sum += {expr};
    }}
    return sum;
}}()"""
        var = expr.limits[0][0]
        start = expr.limits[0][1]
        end = expr.limits[0][2]
        code = template.format(
            dtype=get_type_of_expression(expr.args[0]),
            iterator_dtype='int',
            var=self._print(var),
            start=self._print(start),
            end=self._print(end),
            expr=self._print(expr.function),
            increment=str(1),
            condition=self._print(var) + ' <= ' + self._print(end)  # if start < end else '>='
        )
        return code

    def _print_Product(self, expr):
        template = """[&]() {{
    {dtype} product = ({dtype}) 1;
    for ( {iterator_dtype} {var} = {start}; {condition}; {var} += {increment} ) {{
        product *= {expr};
    }}
    return product;
}}()"""
        var = expr.limits[0][0]
        start = expr.limits[0][1]
        end = expr.limits[0][2]
        code = template.format(
            dtype=get_type_of_expression(expr.args[0]),
            iterator_dtype='int',
            var=self._print(var),
            start=self._print(start),
            end=self._print(end),
            expr=self._print(expr.function),
            increment=str(1),
            condition=self._print(var) + ' <= ' + self._print(end)  # if start < end else '>='
        )
        return code

    def _print_ConditionalFieldAccess(self, node):
        return self._print(sp.Piecewise((node.outofbounds_value, node.outofbounds_condition), (node.access, True)))

    _print_Max = C89CodePrinter._print_Max
    _print_Min = C89CodePrinter._print_Min

    def _print_re(self, expr):
        return f"real({self._print(expr.args[0])})"

    def _print_im(self, expr):
        return f"imag({self._print(expr.args[0])})"

    def _print_ImaginaryUnit(self, expr):
        return "complex<double>{0,1}"

    def _print_TypedImaginaryUnit(self, expr):
        if expr.dtype.numpy_dtype == np.complex64:
            return "complex<float>{0,1}"
        elif expr.dtype.numpy_dtype == np.complex128:
            return "complex<double>{0,1}"
        else:
            raise NotImplementedError(
                "only complex64 and complex128 supported")

    def _print_Complex(self, expr):
        return self._typed_number(expr, np.complex64)


# noinspection PyPep8Naming
class VectorizedCustomSympyPrinter(CustomSympyPrinter):
    SummandInfo = namedtuple("SummandInfo", ['sign', 'term'])

    def __init__(self, instruction_set):
        super(VectorizedCustomSympyPrinter, self).__init__()
        self.instruction_set = instruction_set

    def _scalarFallback(self, func_name, expr, *args, **kwargs):
        expr_type = get_type_of_expression(expr)
        if type(expr_type) is not VectorType:
            return getattr(super(VectorizedCustomSympyPrinter, self), func_name)(expr, *args, **kwargs)
        else:
            assert self.instruction_set['width'] == expr_type.width
            return None

    def _print_Function(self, expr):
        if isinstance(expr, vector_memory_access):
            arg, data_type, aligned, _, mask = expr.args
            instruction = self.instruction_set['loadA'] if aligned else self.instruction_set['loadU']
            return instruction.format("& " + self._print(arg))
        elif isinstance(expr, cast_func):
            arg, data_type = expr.args
            if type(data_type) is VectorType:
                if isinstance(arg, sp.Tuple):
                    is_boolean = get_type_of_expression(arg[0]) == create_type("bool")
                    printed_args = [self._print(a) for a in arg]
                    instruction = 'makeVecBool' if is_boolean else 'makeVec'
                    return self.instruction_set[instruction].format(*printed_args)
                else:
                    is_boolean = get_type_of_expression(arg) == create_type("bool")
                    instruction = 'makeVecConstBool' if is_boolean else 'makeVecConst'
                    return self.instruction_set[instruction].format(self._print(arg))
        elif expr.func == fast_division:
            result = self._scalarFallback('_print_Function', expr)
            if not result:
                result = self.instruction_set['/'].format(self._print(expr.args[0]), self._print(expr.args[1]))
            return result
        elif expr.func == fast_sqrt:
            return "({})".format(self._print(sp.sqrt(expr.args[0])))
        elif expr.func == fast_inv_sqrt:
            result = self._scalarFallback('_print_Function', expr)
            if not result:
                if self.instruction_set['rsqrt']:
                    return self.instruction_set['rsqrt'].format(self._print(expr.args[0]))
                else:
                    return "({})".format(self._print(1 / sp.sqrt(expr.args[0])))
        elif isinstance(expr, vec_any):
            expr_type = get_type_of_expression(expr.args[0])
            if type(expr_type) is not VectorType:
                return self._print(expr.args[0])
            else:
                return self.instruction_set['any'].format(self._print(expr.args[0]))
        elif isinstance(expr, vec_all):
            expr_type = get_type_of_expression(expr.args[0])
            if type(expr_type) is not VectorType:
                return self._print(expr.args[0])
            else:
                return self.instruction_set['all'].format(self._print(expr.args[0]))

        return super(VectorizedCustomSympyPrinter, self)._print_Function(expr)

    def _print_And(self, expr):
        result = self._scalarFallback('_print_And', expr)
        if result:
            return result

        arg_strings = [self._print(a) for a in expr.args]
        assert len(arg_strings) > 0
        result = arg_strings[0]
        for item in arg_strings[1:]:
            result = self.instruction_set['&'].format(result, item)
        return result

    def _print_Or(self, expr):
        result = self._scalarFallback('_print_Or', expr)
        if result:
            return result

        arg_strings = [self._print(a) for a in expr.args]
        assert len(arg_strings) > 0
        result = arg_strings[0]
        for item in arg_strings[1:]:
            result = self.instruction_set['|'].format(result, item)
        return result

    def _print_Add(self, expr, order=None):
        result = self._scalarFallback('_print_Add', expr)
        if result:
            return result

        summands = []
        for term in expr.args:
            if term.func == sp.Mul:
                sign, t = self._print_Mul(term, inside_add=True)
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
            func = self.instruction_set['-'] if summand.sign == -1 else self.instruction_set['+']
            processed = func.format(processed, summand.term)
        return processed

    def _print_Pow(self, expr):
        result = self._scalarFallback('_print_Pow', expr)
        if result:
            return result

        one = self.instruction_set['makeVecConst'].format(1.0)

        if expr.exp.is_integer and expr.exp.is_number and 0 < expr.exp < 8:
            return "(" + self._print(sp.Mul(*[expr.base] * expr.exp, evaluate=False)) + ")"
        elif expr.exp == -1:
            one = self.instruction_set['makeVecConst'].format(1.0)
            return self.instruction_set['/'].format(one, self._print(expr.base))
        elif expr.exp == 0.5:
            return self.instruction_set['sqrt'].format(self._print(expr.base))
        elif expr.exp == -0.5:
            root = self.instruction_set['sqrt'].format(self._print(expr.base))
            return self.instruction_set['/'].format(one, root)
        elif expr.exp.is_integer and expr.exp.is_number and - 8 < expr.exp < 0:
            return self.instruction_set['/'].format(one,
                                                    self._print(sp.Mul(*[expr.base] * (-expr.exp), evaluate=False)))
        else:
            raise ValueError("Generic exponential not supported: " + str(expr))

    def _print_Mul(self, expr, inside_add=False):
        # noinspection PyProtectedMember
        from sympy.core.mul import _keep_coeff

        result = self._scalarFallback('_print_Mul', expr)
        if result:
            return result

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
            result = self.instruction_set['*'].format(result, item)

        if len(b) > 0:
            denominator_str = b_str[0]
            for item in b_str[1:]:
                denominator_str = self.instruction_set['*'].format(denominator_str, item)
            result = self.instruction_set['/'].format(result, denominator_str)

        if inside_add:
            return sign, result
        else:
            if sign < 0:
                return self.instruction_set['*'].format(self._print(S.NegativeOne), result)
            else:
                return result

    def _print_Relational(self, expr):
        result = self._scalarFallback('_print_Relational', expr)
        if result:
            return result
        return self.instruction_set[expr.rel_op].format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_Equality(self, expr):
        result = self._scalarFallback('_print_Equality', expr)
        if result:
            return result
        return self.instruction_set['=='].format(self._print(expr.lhs), self._print(expr.rhs))

    def _print_Piecewise(self, expr):
        result = self._scalarFallback('_print_Piecewise', expr)
        if result:
            return result

        if expr.args[-1].cond.args[0] is not sp.sympify(True):
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")

        result = self._print(expr.args[-1][0])
        for true_expr, condition in reversed(expr.args[:-1]):
            if isinstance(condition, cast_func) and get_type_of_expression(condition.args[0]) == create_type("bool"):
                if not KERNCRAFT_NO_TERNARY_MODE:
                    result = "(({}) ? ({}) : ({}))".format(self._print(condition.args[0]), self._print(true_expr),
                                                           result)
                else:
                    print("Warning - skipping ternary op")
            else:
                # noinspection SpellCheckingInspection
                result = self.instruction_set['blendv'].format(result, self._print(true_expr), self._print(condition))
        return result
