import re
from collections import namedtuple
import hashlib
from typing import Set

import numpy as np
import sympy as sp
from sympy.core import S
from sympy.core.cache import cacheit
from sympy.logic.boolalg import BooleanFalse, BooleanTrue
from sympy.functions.elementary.trigonometric import TrigonometricFunction, InverseTrigonometricFunction
from sympy.functions.elementary.hyperbolic import HyperbolicFunction

from pystencils.astnodes import KernelFunction, LoopOverCoordinate, Node
from pystencils.cpu.vectorization import vec_all, vec_any, CachelineSize
from pystencils.typing import (
    PointerType, VectorType, CastFunc, create_type, get_type_of_expression,
    ReinterpretCastFunc, VectorMemoryAccess, BasicType, TypedSymbol)
from pystencils.enums import Backend
from pystencils.fast_approximation import fast_division, fast_inv_sqrt, fast_sqrt
from pystencils.functions import DivFunc, AddressOf
from pystencils.integer_functions import (
    bit_shift_left, bit_shift_right, bitwise_and, bitwise_or, bitwise_xor,
    int_div, int_power_of_2, modulo_ceil)

try:
    from sympy.printing.c import C99CodePrinter as CCodePrinter  # for sympy versions > 1.6
except ImportError:
    from sympy.printing.ccode import C99CodePrinter as CCodePrinter

__all__ = ['generate_c', 'CustomCodeNode', 'PrintNode', 'get_headers', 'CustomSympyPrinter']


HEADER_REGEX = re.compile(r'^[<"].*[">]$')


def generate_c(ast_node: Node,
               signature_only: bool = False,
               dialect: Backend = Backend.C,
               custom_backend=None,
               with_globals=True) -> str:
    """Prints an abstract syntax tree node as C or CUDA code.

    This function does not need to distinguish for most AST nodes between C, C++ or CUDA code, it just prints 'C-like'
    code as encoded in the abstract syntax tree (AST). The AST is built differently for C or CUDA by calling different
    create_kernel functions.

    Args:
        ast_node: ast representation of kernel
        signature_only: generate signature without function body
        dialect: `Backend`: 'C' or 'CUDA'
        custom_backend: use own custom printer for code generation
        with_globals: enable usage of global variables
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
    elif dialect == Backend.C:
        try:
            # TODO Vectorization Revamp: instruction_set should not be just slapped on ast
            instruction_set = ast_node.instruction_set
        except Exception:
            instruction_set = None
        printer = CBackend(signature_only=signature_only,
                           vector_instruction_set=instruction_set)
    elif dialect == Backend.CUDA:
        from pystencils.backends.cuda_backend import CudaBackend
        printer = CudaBackend(signature_only=signature_only)
    else:
        raise ValueError(f'Unknown {dialect=}')
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

    return headers


# --------------------------------------- Backend Specific Nodes -------------------------------------------------------

# TODO future CustomCodeNode should not be backend specific move it elsewhere
class CustomCodeNode(Node):
    def __init__(self, code, symbols_read, symbols_defined, parent=None):
        super(CustomCodeNode, self).__init__(parent=parent)
        self._code = "\n" + code
        self._symbols_read = set(symbols_read)
        self._symbols_defined = set(symbols_defined)
        self.headers = []

    def get_code(self, dialect, vector_instruction_set, print_arg):
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
        code = f'\nstd::cout << "{symbol_to_print.name}  =  " << {symbol_to_print.name} << std::endl; \n'
        super(PrintNode, self).__init__(code, symbols_read=[symbol_to_print], symbols_defined=set())
        self.headers.append("<iostream>")


class CFunction(TypedSymbol):
    def __new__(cls, function, dtype):
        return CFunction.__xnew_cached_(cls, function, dtype)

    def __new_stage2__(cls, function, dtype):
        return super(CFunction, cls).__xnew__(cls, function, dtype)

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __getnewargs__(self):
        return self.name, self.dtype

    def __getnewargs_ex__(self):
        return (self.name, self.dtype), {}


# ------------------------------------------- Printer ------------------------------------------------------------------


# noinspection PyPep8Naming
class CBackend:

    def __init__(self, sympy_printer=None, signature_only=False, vector_instruction_set=None, dialect=Backend.C):
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
        self._kwargs = {}
        self.sympy_printer._kwargs = self._kwargs

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
            method_name = f"_print_{cls.__name__}"
            if hasattr(self, method_name):
                return getattr(self, method_name)(node)
        raise NotImplementedError(f"{self.__class__.__name__} does not support node of type {node.__class__.__name__}")

    def _print_AbstractType(self, node):
        return str(node)

    def _print_KernelFunction(self, node):
        function_arguments = [f"{self._print(s.symbol.dtype)} {s.symbol.name}" for s in node.get_parameters()
                              if not type(s.symbol) is CFunction]
        launch_bounds = ""
        if self._dialect == Backend.CUDA:
            max_threads = node.indexing.max_threads_per_block()
            if max_threads:
                launch_bounds = f"__launch_bounds__({max_threads}) "
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
        return f"{node.pragma_line}\n{self._print_Block(node)}"

    def _print_LoopOverCoordinate(self, node):
        counter_symbol = node.loop_counter_name
        start = f"int64_t {counter_symbol} = {self.sympy_printer.doprint(node.start)}"
        condition = f"{counter_symbol} < {self.sympy_printer.doprint(node.stop)}"
        update = f"{counter_symbol} += {self.sympy_printer.doprint(node.step)}"
        loop_str = f"for ({start}; {condition}; {update})"
        self._kwargs['loop_counter'] = counter_symbol
        self._kwargs['loop_stop'] = node.stop

        prefix = "\n".join(node.prefix_lines)
        if prefix:
            prefix += "\n"
        return f"{prefix}{loop_str}\n{self._print(node.body)}"

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
            lhs_type = get_type_of_expression(node.lhs)  # TOOD: this should have been typed
            printed_mask = ""
            if type(lhs_type) is VectorType and isinstance(node.lhs, CastFunc):
                arg, data_type, aligned, nontemporal, mask, stride = node.lhs.args
                instr = 'storeU'
                if aligned:
                    instr = 'stream' if nontemporal and 'stream' in self._vector_instruction_set else 'storeA'
                if mask != True:  # NOQA
                    instr = 'maskStoreA' if aligned else 'maskStoreU'
                    if instr not in self._vector_instruction_set:
                        self._vector_instruction_set[instr] = self._vector_instruction_set['store' + instr[-1]].format(
                            '{0}', self._vector_instruction_set['blendv'].format(
                                self._vector_instruction_set['load' + instr[-1]].format('{0}', **self._kwargs),
                                '{1}', '{2}', **self._kwargs), **self._kwargs)
                    printed_mask = self.sympy_printer.doprint(mask)
                    if data_type.base_type.c_name == 'double':
                        if self._vector_instruction_set['double'] == '__m256d':
                            printed_mask = f"_mm256_castpd_si256({printed_mask})"
                        elif self._vector_instruction_set['double'] == '__m128d':
                            printed_mask = f"_mm_castpd_si128({printed_mask})"
                    elif data_type.base_type.c_name == 'float':
                        if self._vector_instruction_set['float'] == '__m256':
                            printed_mask = f"_mm256_castps_si256({printed_mask})"
                        elif self._vector_instruction_set['float'] == '__m128':
                            printed_mask = f"_mm_castps_si128({printed_mask})"

                rhs_type = get_type_of_expression(node.rhs)
                if type(rhs_type) is not VectorType:
                    raise ValueError(f'Cannot vectorize {node.rhs} of type {rhs_type} inside of the pretty printer! '
                                     f'This should have happen earlier!')
                    # rhs = CastFunc(node.rhs, VectorType(rhs_type)) # Unknown width
                else:
                    rhs = node.rhs

                ptr = "&" + self.sympy_printer.doprint(node.lhs.args[0])

                if stride != 1:
                    instr = 'maskStoreS' if mask != True else 'storeS'  # NOQA
                    return self._vector_instruction_set[instr].format(ptr, self.sympy_printer.doprint(rhs),
                                                                      stride, printed_mask, **self._kwargs) + ';'

                pre_code = ''
                if nontemporal and 'cachelineZero' in self._vector_instruction_set:
                    first_cond = f"((uintptr_t) {ptr} & {CachelineSize.mask_symbol}) == 0"
                    offset = sp.Add(*[sp.Symbol(LoopOverCoordinate.get_loop_counter_name(i))
                                      * node.lhs.args[0].field.spatial_strides[i] for i in
                                      range(len(node.lhs.args[0].field.spatial_strides))])
                    if stride == 1:
                        offset = offset.subs({node.lhs.args[0].field.spatial_strides[0]: 1})
                    size = sp.Mul(*node.lhs.args[0].field.spatial_shape)
                    element_size = 8 if data_type.base_type.c_name == 'double' else 4
                    size_cond = f"({offset} + {CachelineSize.symbol/element_size}) < {size}"
                    pre_code = f"if ({first_cond} && {size_cond}) " + "{\n\t" + \
                        self._vector_instruction_set['cachelineZero'].format(ptr, **self._kwargs) + ';\n}\n'

                code = self._vector_instruction_set[instr].format(ptr, self.sympy_printer.doprint(rhs),
                                                                  printed_mask, **self._kwargs) + ';'
                flushcond = f"((uintptr_t) {ptr} & {CachelineSize.mask_symbol}) == {CachelineSize.last_symbol}"
                if nontemporal and 'flushCacheline' in self._vector_instruction_set:
                    code2 = self._vector_instruction_set['flushCacheline'].format(
                        ptr, self.sympy_printer.doprint(rhs), **self._kwargs) + ';'
                    code = f"{code}\nif ({flushcond}) {{\n\t{code2}\n}}"
                elif nontemporal and 'storeAAndFlushCacheline' in self._vector_instruction_set:
                    lhs_hash = hashlib.sha1(self.sympy_printer.doprint(node.lhs).encode('ascii')).hexdigest()[:8]
                    rhs_hash = hashlib.sha1(self.sympy_printer.doprint(rhs).encode('ascii')).hexdigest()[:8]
                    tmpvar = f'_tmp_{lhs_hash}_{rhs_hash}'
                    code = 'const ' + self._print(node.lhs.dtype).replace(' const', '') + ' ' + tmpvar + ' = ' \
                        + self.sympy_printer.doprint(rhs) + ';'
                    code1 = self._vector_instruction_set[instr].format(ptr, tmpvar, printed_mask, **self._kwargs) + ';'
                    code2 = self._vector_instruction_set['storeAAndFlushCacheline'].format(ptr, tmpvar, printed_mask,
                                                                                           **self._kwargs) + ';'
                    code += f"\nif ({flushcond}) {{\n\t{code2}\n}} else {{\n\t{code1}\n}}"
                return pre_code + code
            else:
                return f"{self.sympy_printer.doprint(node.lhs)} = {self.sympy_printer.doprint(node.rhs)};"

    def _print_NontemporalFence(self, _):
        if 'streamFence' in self._vector_instruction_set:
            return self._vector_instruction_set['streamFence'] + ';'
        else:
            return ''

    def _print_CachelineSize(self, node):
        if 'cachelineSize' in self._vector_instruction_set:
            code = f'const size_t {node.symbol} = {self._vector_instruction_set["cachelineSize"]};\n'
            code += f'const size_t {node.mask_symbol} = {node.symbol} - 1;\n'
            vectorsize = self._vector_instruction_set['bytes']
            code += f'const size_t {node.last_symbol} = {node.symbol} - {vectorsize};\n'
            return code
        else:
            return ''

    def _print_TemporaryMemoryAllocation(self, node):
        if self._vector_instruction_set:
            align = self._vector_instruction_set['bytes']
        else:
            align = node.symbol.dtype.base_type.numpy_dtype.itemsize

        np_dtype = node.symbol.dtype.base_type.numpy_dtype
        required_size = np_dtype.itemsize * node.size + align
        size = modulo_ceil(required_size, align)
        code = "#if defined(_MSC_VER)\n"
        code += "{dtype} {name}=({dtype})_aligned_malloc({size}, {align}) + {offset};\n"
        code += "#elif __cplusplus >= 201703L || __STDC_VERSION__ >= 201112L\n"
        code += "{dtype} {name}=({dtype})aligned_alloc({align}, {size}) + {offset};\n"
        code += "#else\n"
        code += "{dtype} {name};\n"
        code += "posix_memalign((void**) &{name}, {align}, {size});\n"
        code += "{name} += {offset};\n"
        code += "#endif"
        return code.format(dtype=node.symbol.dtype,
                           name=self.sympy_printer.doprint(node.symbol.name),
                           size=self.sympy_printer.doprint(size),
                           offset=int(node.offset(align)),
                           align=align)

    def _print_TemporaryMemoryFree(self, node):
        if self._vector_instruction_set:
            align = self._vector_instruction_set['bytes']
        else:
            align = node.symbol.dtype.base_type.numpy_dtype.itemsize

        code = "#if defined(_MSC_VER)\n"
        code += "_aligned_free(%s - %d);\n" % (self.sympy_printer.doprint(node.symbol.name), node.offset(align))
        code += "#else\n"
        code += "free(%s - %d);\n" % (self.sympy_printer.doprint(node.symbol.name), node.offset(align))
        code += "#endif"
        return code

    def _print_SkipIteration(self, _):
        return "continue;"

    def _print_CustomCodeNode(self, node):
        return node.get_code(self._dialect, self._vector_instruction_set, print_arg=self.sympy_printer._print)

    def _print_SourceCodeComment(self, node):
        return f"/* {node.text } */"

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
        result = f"if ({condition_expr})\n{true_block} "
        if node.false_block:
            false_block = self._print_Block(node.false_block)
            result += f"else {false_block}"
        return result


# ------------------------------------------ Helper function & classes -------------------------------------------------


# noinspection PyPep8Naming
class CustomSympyPrinter(CCodePrinter):

    def __init__(self):
        super(CustomSympyPrinter, self).__init__()

    def _print_Pow(self, expr):
        """Don't use std::pow function, for small integer exponents, write as multiplication"""
        if isinstance(expr.exp, sp.Integer) and (-8 < expr.exp < 8):
            raise ValueError(f"This expression: {expr} contains a pow function that should be simplified already with "
                             f"a sequence of multiplications")
        return super(CustomSympyPrinter, self)._print_Pow(expr)

    # TODO don't print ones in sp.Mul

    def _print_Rational(self, expr):
        """Evaluate all rationals i.e. print 0.25 instead of 1.0/4.0"""
        res = str(expr.evalf(17))
        return res

    def _print_Equality(self, expr):
        """Equality operator is not printable in default printer"""
        return '((' + self._print(expr.lhs) + ") == (" + self._print(expr.rhs) + '))'

    def _print_Piecewise(self, expr):
        """Print piecewise in one line (remove newlines)"""
        result = super(CustomSympyPrinter, self)._print_Piecewise(expr)
        return result.replace("\n", "")

    def _print_Abs(self, expr):
        if expr.args[0].is_integer:
            return f'abs({self._print(expr.args[0])})'
        else:
            return f'fabs({self._print(expr.args[0])})'

    def _print_AbstractType(self, node):
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
        if isinstance(expr, ReinterpretCastFunc):
            arg, data_type = expr.args
            return f"*(({self._print(PointerType(data_type, restrict=False))})(& {self._print(arg)}))"
        elif isinstance(expr, AddressOf):
            assert len(expr.args) == 1, "address_of must only have one argument"
            return f"&({self._print(expr.args[0])})"
        elif isinstance(expr, CastFunc):
            arg, data_type = expr.args
            if arg.is_Number and not isinstance(arg, (sp.core.numbers.Infinity, sp.core.numbers.NegativeInfinity)):
                return self._typed_number(arg, data_type)
            elif isinstance(arg, (InverseTrigonometricFunction, TrigonometricFunction, HyperbolicFunction)) \
                    and data_type == BasicType('float32'):
                known = self.known_functions[arg.__class__.__name__.lower()]
                code = self._print(arg)
                return code.replace(known, f"{known}f")
            elif isinstance(arg, (sp.Pow, sp.exp)) and data_type == BasicType('float32'):
                known = ['sqrt', 'cbrt', 'pow', 'exp']
                code = self._print(arg)
                for k in known:
                    if k in code:
                        return code.replace(k, f'{k}f')
                raise ValueError(f"{code} doesn't give {known=} function back.")
            else:
                return f"(({data_type})({self._print(arg)}))"
        elif isinstance(expr, fast_division):
            raise ValueError("fast_division is only supported for Taget.GPU")
        elif isinstance(expr, fast_sqrt):
            raise ValueError("fast_sqrt is only supported for Taget.GPU")
        elif isinstance(expr, fast_inv_sqrt):
            raise ValueError("fast_inv_sqrt is only supported for Taget.GPU")
        elif isinstance(expr, vec_any) or isinstance(expr, vec_all):
            return self._print(expr.args[0])
        elif isinstance(expr, sp.Abs):
            return f"abs({self._print(expr.args[0])})"
        elif isinstance(expr, sp.Mod):
            if expr.args[0].is_integer and expr.args[1].is_integer:
                return f"({self._print(expr.args[0])} % {self._print(expr.args[1])})"
            else:
                return f"fmod({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        elif expr.func in infix_functions:
            return f"({self._print(expr.args[0])} {infix_functions[expr.func]} {self._print(expr.args[1])})"
        elif expr.func == int_power_of_2:
            return f"(1 << ({self._print(expr.args[0])}))"
        elif expr.func == int_div:
            return f"(({self._print(expr.args[0])}) / ({self._print(expr.args[1])}))"
        elif expr.func == DivFunc:
            return f'(({self._print(expr.divisor)}) / ({self._print(expr.dividend)}))'
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
        elif dtype.is_int():
            tokens = res.split('.')
            if len(tokens) == 1: 
                return res
            elif int(tokens[1]) != 0:
                raise ValueError(f"Cannot print non-integer number {res} as an integer.")
            else:
                return tokens[0]
        else:
            return res

    def _print_ConditionalFieldAccess(self, node):
        return self._print(sp.Piecewise((node.outofbounds_value, node.outofbounds_condition), (node.access, True)))

    def _print_Max(self, expr):
        def inner_print_max(args):
            if len(args) == 1:
                return self._print(args[0])
            half = len(args) // 2
            a = inner_print_max(args[:half])
            b = inner_print_max(args[half:])
            return f"(({a} > {b}) ? {a} : {b})"
        return inner_print_max(expr.args)

    def _print_Min(self, expr):
        def inner_print_min(args):
            if len(args) == 1:
                return self._print(args[0])
            half = len(args) // 2
            a = inner_print_min(args[:half])
            b = inner_print_min(args[half:])
            return f"(({a} < {b}) ? {a} : {b})"
        return inner_print_min(expr.args)


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

    def _print_Abs(self, expr):
        if 'abs' in self.instruction_set and isinstance(expr.args[0], VectorMemoryAccess):
            return self.instruction_set['abs'].format(self._print(expr.args[0]), **self._kwargs)
        return super()._print_Abs(expr)

    def _typed_vectorized_number(self, expr, data_type):
        basic_data_type = data_type.base_type
        number = self._typed_number(expr, basic_data_type)
        instruction = 'makeVecConst'
        if basic_data_type.is_bool():
            instruction = 'makeVecConstBool'
        # TODO Vectorization Revamp: is int, or sint, or uint (my guess is sint)
        elif basic_data_type.is_int():
            instruction = 'makeVecConstInt'
        return self.instruction_set[instruction].format(number, **self._kwargs)

    def _typed_vectorized_symbol(self, expr, data_type):
        if not isinstance(expr, TypedSymbol):
            raise ValueError(f'{expr} is not a TypeSymbol. It is {expr.type=}')
        basic_data_type = data_type.base_type
        symbol = self._print(expr)
        if basic_data_type != expr.dtype:
            symbol = f'(({basic_data_type})({symbol}))'

        instruction = 'makeVecConst'
        if basic_data_type.is_bool():
            instruction = 'makeVecConstBool'
        # TODO Vectorization Revamp: is int, or sint, or uint (my guess is sint)
        elif basic_data_type.is_int():
            instruction = 'makeVecConstInt'
        return self.instruction_set[instruction].format(symbol, **self._kwargs)

    def _print_CastFunc(self, expr):
        arg, data_type = expr.args
        if type(data_type) is VectorType:
            base_type = data_type.base_type
            # vector_memory_access is a cast_func itself so it should't be directly inside a cast_func
            assert not isinstance(arg, VectorMemoryAccess)
            if isinstance(arg, sp.Tuple):
                is_boolean = get_type_of_expression(arg[0]) == create_type("bool")
                is_integer = get_type_of_expression(arg[0]) == create_type("int")
                printed_args = [self._print(a) for a in arg]
                instruction = 'makeVecBool' if is_boolean else 'makeVecInt' if is_integer else 'makeVec'
                if instruction == 'makeVecInt' and 'makeVecIndex' in self.instruction_set:
                    increments = np.array(arg)[1:] - np.array(arg)[:-1]
                    if len(set(increments)) == 1:
                        return self.instruction_set['makeVecIndex'].format(printed_args[0], increments[0],
                                                                           **self._kwargs)
                return self.instruction_set[instruction].format(*printed_args, **self._kwargs)
            else:
                if arg.is_Number and not isinstance(arg, (sp.core.numbers.Infinity, sp.core.numbers.NegativeInfinity)):
                    return self._typed_vectorized_number(arg, data_type)
                elif isinstance(arg, TypedSymbol):
                    return self._typed_vectorized_symbol(arg, data_type)
                elif isinstance(arg, (InverseTrigonometricFunction, TrigonometricFunction, HyperbolicFunction)) \
                        and base_type == BasicType('float32'):
                    raise NotImplementedError('Vectorizer is not tested for trigonometric functions yet')
                    # known = self.known_functions[arg.__class__.__name__.lower()]
                    # code = self._print(arg)
                    # return code.replace(known, f"{known}f")
                elif isinstance(arg, sp.Pow):
                    if base_type == BasicType('float32') or base_type == BasicType('float64'):
                        return self._print_Pow(arg)
                    else:
                        raise NotImplementedError('Integer Pow is not implemented')
                elif isinstance(arg, sp.UnevaluatedExpr):
                    return self._print(arg.args[0])
                else:
                    raise NotImplementedError('Vectorizer cannot cast between different datatypes')
                    # to_type = self.instruction_set['suffix'][data_type.base_type.c_name]
                    # from_type = self.instruction_set['suffix'][get_type_of_expression(arg).base_type.c_name]
                    # return self.instruction_set['cast'].format(from_type, to_type, self._print(arg))
        else:
            return self._scalarFallback('_print_Function', expr)
            # raise ValueError(f'Non VectorType cast "{data_type}" in vectorized code.')

    def _print_Function(self, expr):
        if isinstance(expr, VectorMemoryAccess):
            arg, data_type, aligned, _, mask, stride = expr.args
            if stride != 1:
                return self.instruction_set['loadS'].format(f"& {self._print(arg)}", stride, **self._kwargs)
            instruction = self.instruction_set['loadA'] if aligned else self.instruction_set['loadU']
            return instruction.format(f"& {self._print(arg)}", **self._kwargs)
        elif expr.func == DivFunc:
            result = self._scalarFallback('_print_Function', expr)
            if not result:
                result = self.instruction_set['/'].format(self._print(expr.divisor), self._print(expr.dividend),
                                                          **self._kwargs)
            return result
        elif isinstance(expr, fast_division):
            raise ValueError("fast_division is only supported for Taget.GPU")
        elif isinstance(expr, fast_sqrt):
            raise ValueError("fast_sqrt is only supported for Taget.GPU")
        elif isinstance(expr, fast_inv_sqrt):
            raise ValueError("fast_inv_sqrt is only supported for Taget.GPU")
        elif isinstance(expr, vec_any) or isinstance(expr, vec_all):
            instr = 'any' if isinstance(expr, vec_any) else 'all'
            expr_type = get_type_of_expression(expr.args[0])
            if type(expr_type) is not VectorType:
                return self._print(expr.args[0])
            else:
                if isinstance(expr.args[0], sp.Rel):
                    op = expr.args[0].rel_op
                    if (instr, op) in self.instruction_set:
                        return self.instruction_set[(instr, op)].format(*[self._print(a) for a in expr.args[0].args],
                                                                        **self._kwargs)
                return self.instruction_set[instr].format(self._print(expr.args[0]), **self._kwargs)

        return super(VectorizedCustomSympyPrinter, self)._print_Function(expr)

    def _print_And(self, expr):
        result = self._scalarFallback('_print_And', expr)
        if result:
            return result

        arg_strings = [self._print(a) for a in expr.args]
        assert len(arg_strings) > 0
        result = arg_strings[0]
        for item in arg_strings[1:]:
            result = self.instruction_set['&'].format(result, item, **self._kwargs)
        return result

    def _print_Or(self, expr):
        result = self._scalarFallback('_print_Or', expr)
        if result:
            return result

        arg_strings = [self._print(a) for a in expr.args]
        assert len(arg_strings) > 0
        result = arg_strings[0]
        for item in arg_strings[1:]:
            result = self.instruction_set['|'].format(result, item, **self._kwargs)
        return result

    def _print_Add(self, expr, order=None):
        try:
            result = self._scalarFallback('_print_Add', expr)
        except Exception:
            result = None
        if result:
            return result
        args = expr.args

        # special treatment for all-integer args, for loop index arithmetic until we have proper int vectorization
        suffix = ""
        if all([(type(e) is CastFunc and str(e.dtype) == self.instruction_set['int']) or isinstance(e, sp.Integer)
                or (type(e) is TypedSymbol and isinstance(e.dtype, BasicType) and e.dtype.is_int()) for e in args]):
            dtype = set([e.dtype for e in args if type(e) is CastFunc])
            assert len(dtype) == 1
            dtype = dtype.pop()
            args = [CastFunc(e, dtype) if (isinstance(e, sp.Integer) or isinstance(e, TypedSymbol)) else e
                    for e in args]
            suffix = "int"

        summands = []
        for term in args:
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
            func = self.instruction_set['-' + suffix] if summand.sign == -1 else self.instruction_set['+' + suffix]
            processed = func.format(processed, summand.term, **self._kwargs)
        return processed

    def _print_Pow(self, expr):
        # Due to loop cutting sp.Mul is evaluated again.

        try:
            result = self._scalarFallback('_print_Pow', expr)
        except ValueError:
            result = None
        if result:
            return result

        one = self.instruction_set['makeVecConst'].format(1.0, **self._kwargs)
        root = self.instruction_set['sqrt'].format(self._print(expr.base), **self._kwargs)

        if isinstance(expr.exp, CastFunc) and expr.exp.args[0].is_number:
            exp = expr.exp.args[0]
        else:
            exp = expr.exp

        # TODO the printer should not have any intelligence like this.
        # TODO To remove all of these cases the vectoriser needs to be reworked. See loop cutting
        if exp.is_integer and exp.is_number and 0 < exp < 8:
            return self._print(sp.Mul(*[expr.base] * exp, evaluate=False))
        elif exp == 0.5:
            return root
        elif exp == -0.5:
            return self.instruction_set['/'].format(one, root, **self._kwargs)
        else:
            raise ValueError("Generic exponential not supported: " + str(expr))

    def _print_Mul(self, expr, inside_add=False):
        # noinspection PyProtectedMember
        from sympy.core.mul import _keep_coeff

        if not inside_add:
            result = self._scalarFallback('_print_Mul', expr)
        else:
            result = None
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
            result = self.instruction_set['*'].format(result, item, **self._kwargs)

        if len(b) > 0:
            denominator_str = b_str[0]
            for item in b_str[1:]:
                denominator_str = self.instruction_set['*'].format(denominator_str, item, **self._kwargs)
            result = self.instruction_set['/'].format(result, denominator_str, **self._kwargs)

        if inside_add:
            return sign, result
        else:
            if sign < 0:
                return self.instruction_set['*'].format(self._print(S.NegativeOne), result, **self._kwargs)
            else:
                return result

    def _print_Relational(self, expr):
        result = self._scalarFallback('_print_Relational', expr)
        if result:
            return result
        return self.instruction_set[expr.rel_op].format(self._print(expr.lhs), self._print(expr.rhs), **self._kwargs)

    def _print_Equality(self, expr):
        result = self._scalarFallback('_print_Equality', expr)
        if result:
            return result
        return self.instruction_set['=='].format(self._print(expr.lhs), self._print(expr.rhs), **self._kwargs)

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
            if isinstance(condition, CastFunc) and get_type_of_expression(condition.args[0]) == create_type("bool"):
                result = "(({}) ? ({}) : ({}))".format(self._print(condition.args[0]), self._print(true_expr),
                                                       result, **self._kwargs)
            else:
                # noinspection SpellCheckingInspection
                result = self.instruction_set['blendv'].format(result, self._print(true_expr), self._print(condition),
                                                               **self._kwargs)
        return result
