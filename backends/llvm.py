import llvmlite.ir as ir
import functools

from sympy.printing.printer import Printer
from sympy import S
# S is numbers?

from pystencils.llvm.control_flow import Loop
from ..types import createType
from ..astnodes import Indexed


def generateLLVM(ast_node, module=ir.Module(), builder=ir.IRBuilder()):
    """
    Prints the ast as llvm code
    """
    printer = LLVMPrinter(module, builder)
    return printer._print(ast_node)


class LLVMPrinter(Printer):
    """Convert expressions to LLVM IR"""
    def __init__(self, module, builder, fn=None, *args, **kwargs):
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

    def _add_tmp_var(self, name, value):
        self.tmp_var[name] = value

    def _print_Number(self, n):
        if n.dtype == createType("int"):
            return ir.Constant(self.integer, int(n))
        elif n.dtype == createType("double"):
            return ir.Constant(self.fp_type, float(n))
        else:
            raise NotImplementedError("Numbers can only have int and double", n)

    def _print_Float(self, expr):
        return ir.Constant(self.fp_type, expr.p)

    def _print_Integer(self, expr):
        return ir.Constant(self.integer, expr.p)

    def _print_int(self, i):
        return ir.Constant(self.integer, i)

    def _print_Symbol(self, s):
        val = self.tmp_var.get(s)
        if not val:
            # look up parameter with name s
            val = self.func_arg_map.get(s.name)
        if not val:
            raise LookupError("Symbol not found: %s" % s)
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
        if expr.dtype == createType('double'):
            mul = self.builder.fmul
        else: # int TODO others?
            mul = self.builder.mul
        for node in nodes[1:]:
            e = mul(e, node)
        return e

    def _print_Add(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        if expr.dtype == createType('double'):
            add = self.builder.fadd
        else: # int TODO others?
            add = self.builder.add
        for node in nodes[1:]:
            e = add(e, node)
        return e

    def _print_KernelFunction(self, function):
        return_type = self.void
        # TODO argument in their own call? -> nope
        parameter_type = []
        for parameter in function.parameters:
            # TODO what about ptr shape and stride argument?
            if parameter.isFieldArgument:
                parameter_type.append(self.fp_pointer)
            else:
                parameter_type.append(self.fp_type)
        func_type = ir.FunctionType(return_type, tuple(parameter_type))
        name = function.functionName
        fn = ir.Function(self.module, func_type, name)
        self.ext_fn[name] = fn

        # set proper names to arguments
        for i, arg in enumerate(fn.args):
            arg.name = function.parameters[i].name
            self.func_arg_map[function.parameters[i].name] = arg

        # func.attributes.add("inlinehint")
        # func.attributes.add("argmemonly")
        block = fn.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self._print(function.body)
        self.builder.ret_void()
        self.fn = fn
        return fn

    def _print_Block(self, block):
        for node in block.args:
            self._print(node)

    def _print_LoopOverCoordinate(self, loop):
        with Loop(self.builder, self._print(loop.start), self._print(loop.stop), self._print(loop.step),
                  loop.loopCounterName, loop.loopCounterSymbol.name) as i:
            self._add_tmp_var(loop.loopCounterSymbol, i)
            # TODO remove tmp var
            self._print(loop.body)

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

    def _print_Conversion(self, conversion):
        node = self._print(conversion.args[0])
        to_dtype = conversion.dtype
        from_dtype = conversion.args[0].dtype
        # (From, to)
        decision = {
            (createType("int"), createType("double")): functools.partial(self.builder.sitofp, node, self.fp_type),
            (createType("double"), createType("int")): functools.partial(self.builder.fptosi, node, self.integer),
            (createType("double *"), createType("int")): functools.partial(self.builder.ptrtoint, node, self.integer),
            (createType("int"), createType("double *")): functools.partial(self.builder.inttoptr, node, self.fp_pointer),
            (createType("double * restrict"), createType("int")): functools.partial(self.builder.ptrtoint, node, self.integer),
            (createType("int"), createType("double * restrict")): functools.partial(self.builder.inttoptr, node, self.fp_pointer),
            (createType("double * restrict const"), createType("int")): functools.partial(self.builder.ptrtoint, node, self.integer),
            (createType("int"), createType("double * restrict const")): functools.partial(self.builder.inttoptr, node, self.fp_pointer),
            }
        # TODO float, const, restrict
        # TODO bitcast, addrspacecast
        # print([x for x in decision.keys()])
        # print("Types:")
        # print([(type(x), type(y)) for (x, y) in decision.keys()])
        # print("Cast:")
        # print((from_dtype, to_dtype))
        return decision[(from_dtype, to_dtype)]()

    def _print_Indexed(self, indexed):
        ptr = self._print(indexed.base.label)
        index = self._print(indexed.args[1])
        gep = self.builder.gep(ptr, [index])
        return self.builder.load(gep, name=indexed.base.label.name)

    def _print_PointerArithmetic(self, pointer):
        ptr = self._print(pointer.pointer)
        index = self._print(pointer.offset)
        return self.builder.gep(ptr, [index])

    # Should have a list of math library functions to validate this.
    # TODO function calls
    def _print_Function(self, expr):
        name = expr.func.__name__
        e0 = self._print(expr.args[0])
        fn = self.ext_fn.get(name)
        if not fn:
            fn_type = ir.FunctionType(self.fp_type, [self.fp_type])
            fn = ir.Function(self.module, fn_type, name)
            self.ext_fn[name] = fn
        return self.builder.call(fn, [e0], name)

    def emptyPrinter(self, expr):
        raise TypeError("Unsupported type for LLVM JIT conversion: %s %s"
                        % (type(expr), expr))
