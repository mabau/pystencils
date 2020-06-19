import functools

import llvmlite.ir as ir
import llvmlite.llvmpy.core as lc
import sympy as sp
from sympy import Indexed, S
from sympy.printing.printer import Printer

from pystencils.assignment import Assignment
from pystencils.data_types import (
    collate_types, create_composite_type_from_string, create_type, get_type_of_expression,
    to_llvm_type)
from pystencils.llvm.control_flow import Loop


# From Numba
def set_cuda_kernel(lfunc):
    from llvmlite.llvmpy.core import MetaData, MetaDataString, Constant, Type

    m = lfunc.module

    ops = lfunc, MetaDataString.get(m, "kernel"), Constant.int(Type.int(), 1)
    md = MetaData.get(m, ops)

    nmd = m.get_or_insert_named_metadata('nvvm.annotations')
    nmd.add(md)

    # set nvvm ir version
    i32 = ir.IntType(32)
    md_ver = m.add_metadata([i32(1), i32(2), i32(2), i32(0)])
    m.add_named_metadata('nvvmir.version', md_ver)


# From Numba
def _call_sreg(builder, name):
    module = builder.module
    fnty = lc.Type.function(lc.Type.int(), ())
    fn = module.get_or_insert_function(fnty, name=name)
    return builder.call(fn, ())


def generate_llvm(ast_node, module=None, builder=None, target='cpu'):
    """Prints the ast as llvm code."""
    if module is None:
        module = lc.Module()
    if builder is None:
        builder = ir.IRBuilder()
    printer = LLVMPrinter(module, builder, target=target)
    return printer._print(ast_node)


# noinspection PyPep8Naming
class LLVMPrinter(Printer):
    """Convert expressions to LLVM IR"""

    def __init__(self, module, builder, fn=None, target='cpu', *args, **kwargs):
        self.func_arg_map = kwargs.pop("func_arg_map", {})
        super(LLVMPrinter, self).__init__(*args, **kwargs)
        self.fp_type = ir.DoubleType()
        self.fp_pointer = self.fp_type.as_pointer()
        self.integer = ir.IntType(64)
        self.integer_pointer = self.integer.as_pointer()
        self.void = ir.VoidType()
        self.module = module
        self.builder = builder
        self.fn = fn
        self.ext_fn = {}  # keep track of wrappers to external functions
        self.tmp_var = {}
        self.target = target

    def _add_tmp_var(self, name, value):
        self.tmp_var[name] = value

    def _remove_tmp_var(self, name):
        del self.tmp_var[name]

    def _print_Number(self, n):
        if get_type_of_expression(n) == create_type("int"):
            return ir.Constant(self.integer, int(n))
        elif get_type_of_expression(n) == create_type("double"):
            return ir.Constant(self.fp_type, float(n))
        else:
            raise NotImplementedError("Numbers can only have int and double", n)

    def _print_Float(self, expr):
        return ir.Constant(self.fp_type, float(expr))

    def _print_Integer(self, expr):
        return ir.Constant(self.integer, int(expr))

    def _print_int(self, i):
        return ir.Constant(self.integer, i)

    def _print_Symbol(self, s):
        val = self.tmp_var.get(s)
        if not val:
            # look up parameter with name s
            val = self.func_arg_map.get(s.name)
        if not val:
            raise LookupError(f"Symbol not found: {s}")
        return val

    def _print_Pow(self, expr):
        base0 = self._print(expr.base)
        if expr.exp == S.NegativeOne:
            return self.builder.fdiv(ir.Constant(self.fp_type, 1.0), base0)
        if expr.exp == S.Half:
            fn = self.ext_fn.get("sqrt")
            if not fn:
                fn_type = ir.FunctionType(self.fp_type, [self.fp_type])
                fn = ir.Function(self.module, fn_type, "sqrt")
                self.ext_fn["sqrt"] = fn
            return self.builder.call(fn, [base0], "sqrt")
        if expr.exp == 2:
            return self.builder.fmul(base0, base0)
        elif expr.exp == 3:
            return self.builder.fmul(self.builder.fmul(base0, base0), base0)

        exp0 = self._print(expr.exp)
        fn = self.ext_fn.get("pow")
        if not fn:
            fn_type = ir.FunctionType(self.fp_type, [self.fp_type, self.fp_type])
            fn = ir.Function(self.module, fn_type, "pow")
            self.ext_fn["pow"] = fn
        return self.builder.call(fn, [base0, exp0], "pow")

    def _print_Mul(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        if get_type_of_expression(expr) == create_type('double'):
            mul = self.builder.fmul
        else:  # int TODO unsigned/signed
            mul = self.builder.mul
        for node in nodes[1:]:
            e = mul(e, node)
        return e

    def _print_Add(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        if get_type_of_expression(expr) == create_type('double'):
            add = self.builder.fadd
        else:  # int TODO unsigned/signed
            add = self.builder.add
        for node in nodes[1:]:
            e = add(e, node)
        return e

    def _print_Or(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.or_(e, node)
        return e

    def _print_And(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.and_(e, node)
        return e

    def _print_StrictLessThan(self, expr):
        return self._comparison('<', expr)

    def _print_LessThan(self, expr):
        return self._comparison('<=', expr)

    def _print_StrictGreaterThan(self, expr):
        return self._comparison('>', expr)

    def _print_GreaterThan(self, expr):
        return self._comparison('>=', expr)

    def _print_Unequality(self, expr):
        return self._comparison('!=', expr)

    def _print_Equality(self, expr):
        return self._comparison('==', expr)

    def _comparison(self, cmpop, expr):
        if collate_types([get_type_of_expression(arg) for arg in expr.args]) == create_type('double'):
            comparison = self.builder.fcmp_unordered
        else:
            comparison = self.builder.icmp_signed
        return comparison(cmpop, self._print(expr.lhs), self._print(expr.rhs))

    def _print_KernelFunction(self, func):
        # KernelFunction does not posses a return type
        return_type = self.void
        parameter_type = []
        parameters = func.get_parameters()
        for parameter in parameters:
            parameter_type.append(to_llvm_type(parameter.symbol.dtype, nvvm_target=self.target == 'gpu'))
        func_type = ir.FunctionType(return_type, tuple(parameter_type))
        name = func.function_name
        fn = ir.Function(self.module, func_type, name)
        self.ext_fn[name] = fn

        # set proper names to arguments
        for i, arg in enumerate(fn.args):
            arg.name = parameters[i].symbol.name
            self.func_arg_map[parameters[i].symbol.name] = arg

        # func.attributes.add("inlinehint")
        # func.attributes.add("argmemonly")
        block = fn.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)  # TODO use goto_block instead
        self._print(func.body)
        self.builder.ret_void()
        self.fn = fn
        if self.target == 'gpu':
            set_cuda_kernel(fn)

        return fn

    def _print_Block(self, block):
        for node in block.args:
            self._print(node)

    def _print_LoopOverCoordinate(self, loop):
        with Loop(self.builder, self._print(loop.start), self._print(loop.stop), self._print(loop.step),
                  loop.loop_counter_name, loop.loop_counter_symbol.name) as i:
            self._add_tmp_var(loop.loop_counter_symbol, i)
            self._print(loop.body)
            self._remove_tmp_var(loop.loop_counter_symbol)

    def _print_SympyAssignment(self, assignment):
        expr = self._print(assignment.rhs)
        lhs = assignment.lhs
        if isinstance(lhs, Indexed):
            ptr = self._print(lhs.base.label)
            index = self._print(lhs.args[1])
            gep = self.builder.gep(ptr, [index])
            return self.builder.store(expr, gep)
        self.func_arg_map[assignment.lhs.name] = expr
        return expr

    def _print_boolean_cast_func(self, conversion):
        return self._print_cast_func(conversion)

    def _print_cast_func(self, conversion):
        node = self._print(conversion.args[0])
        to_dtype = get_type_of_expression(conversion)
        from_dtype = get_type_of_expression(conversion.args[0])
        if from_dtype == to_dtype:
            return self._print(conversion.args[0])

        # (From, to)
        decision = {
            (create_composite_type_from_string("int32"),
             create_composite_type_from_string("int64")): functools.partial(self.builder.zext, node, self.integer),
            (create_composite_type_from_string("int16"),
             create_composite_type_from_string("int64")): functools.partial(self.builder.zext, node, self.integer),
            (create_composite_type_from_string("int"),
             create_composite_type_from_string("double")): functools.partial(self.builder.sitofp, node, self.fp_type),
            (create_composite_type_from_string("int16"),
             create_composite_type_from_string("double")): functools.partial(self.builder.sitofp, node, self.fp_type),
            (create_composite_type_from_string("double"),
             create_composite_type_from_string("int")): functools.partial(self.builder.fptosi, node, self.integer),
            (create_composite_type_from_string("double *"),
             create_composite_type_from_string("int")): functools.partial(self.builder.ptrtoint, node, self.integer),
            (create_composite_type_from_string("int"),
             create_composite_type_from_string("double *")): functools.partial(self.builder.inttoptr,
                                                                               node, self.fp_pointer),
            (create_composite_type_from_string("double * restrict"),
             create_composite_type_from_string("int")): functools.partial(self.builder.ptrtoint, node, self.integer),
            (create_composite_type_from_string("int"),
             create_composite_type_from_string("double * restrict")): functools.partial(self.builder.inttoptr, node,
                                                                                        self.fp_pointer),
            (create_composite_type_from_string("double * restrict const"),
             create_composite_type_from_string("int")): functools.partial(self.builder.ptrtoint, node,
                                                                          self.integer),
            (create_composite_type_from_string("int"),
             create_composite_type_from_string("double * restrict const")): functools.partial(self.builder.inttoptr,
                                                                                              node, self.fp_pointer),
        }
        # TODO float, TEST: const, restrict
        # TODO bitcast, addrspacecast
        # TODO unsigned/signed fills
        # print([x for x in decision.keys()])
        # print("Types:")
        # print([(type(x), type(y)) for (x, y) in decision.keys()])
        # print("Cast:")
        # print((from_dtype, to_dtype))
        return decision[(from_dtype, to_dtype)]()

    def _print_pointer_arithmetic_func(self, pointer):
        ptr = self._print(pointer.args[0])
        index = self._print(pointer.args[1])
        return self.builder.gep(ptr, [index])

    def _print_Indexed(self, indexed):
        ptr = self._print(indexed.base.label)
        index = self._print(indexed.args[1])
        gep = self.builder.gep(ptr, [index])
        return self.builder.load(gep, name=indexed.base.label.name)

    def _print_Piecewise(self, piece):
        if not piece.args[-1].cond:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        if piece.has(Assignment):
            raise NotImplementedError('The llvm-backend does not support assignments'
                                      'in the Piecewise function. It is questionable'
                                      'whether to implement it. So far there is no'
                                      'use-case to test it.')
        else:
            phi_data = []
            after_block = self.builder.append_basic_block()
            for (expr, condition) in piece.args:
                if condition == sp.sympify(True):  # Don't use 'is' use '=='!
                    phi_data.append((self._print(expr), self.builder.block))
                    self.builder.branch(after_block)
                    self.builder.position_at_end(after_block)
                else:
                    cond = self._print(condition)
                    true_block = self.builder.append_basic_block()
                    false_block = self.builder.append_basic_block()
                    self.builder.cbranch(cond, true_block, false_block)
                    self.builder.position_at_end(true_block)
                    phi_data.append((self._print(expr), true_block))
                    self.builder.branch(after_block)
                    self.builder.position_at_end(false_block)

            phi = self.builder.phi(to_llvm_type(get_type_of_expression(piece), nvvm_target=self.target == 'gpu'))
            for (val, block) in phi_data:
                phi.add_incoming(val, block)
            return phi

    def _print_Conditional(self, node):
        cond = self._print(node.condition_expr)
        with self.builder.if_else(cond) as (then, otherwise):
            with then:
                self._print(node.true_block)       # emit instructions for when the predicate is true
            with otherwise:
                self._print(node.false_block)       # emit instructions for when the predicate is true

        # No return!

    def _print_Function(self, expr):
        name = expr.func.__name__
        e0 = self._print(expr.args[0])
        fn = self.ext_fn.get(name)
        if not fn:
            fn_type = ir.FunctionType(self.fp_type, [self.fp_type])
            fn = ir.Function(self.module, fn_type, name)
            self.ext_fn[name] = fn
        return self.builder.call(fn, [e0], name)

    def empty_printer(self, expr):
        try:
            import inspect
            mro = inspect.getmro(expr)
        except AttributeError:
            mro = "None"
        raise TypeError("Unsupported type for LLVM JIT conversion: Expression:\"%s\", Type:\"%s\", MRO:%s"
                        % (expr, type(expr), mro))

    # from: https://llvm.org/docs/NVPTXUsage.html#nvptx-intrinsics
    INDEXING_FUNCTION_MAPPING = {
        'blockIdx': 'llvm.nvvm.read.ptx.sreg.ctaid',
        'threadIdx': 'llvm.nvvm.read.ptx.sreg.tid',
        'blockDim': 'llvm.nvvm.read.ptx.sreg.ntid',
        'gridDim': 'llvm.nvvm.read.ptx.sreg.nctaid'
    }

    def _print_ThreadIndexingSymbol(self, node):
        symbol_name: str = node.name
        function_name, dimension = tuple(symbol_name.split("."))
        function_name = self.INDEXING_FUNCTION_MAPPING[function_name]
        name = f"{function_name}.{dimension}"

        return self.builder.zext(_call_sreg(self.builder, name), self.integer)
