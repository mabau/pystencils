import llvmlite.ir as ir
import llvmlite.binding as llvm
import logging.config

from sympy.utilities.codegen import CCodePrinter
from pystencils.ast import Node

from sympy.printing.printer import Printer
from sympy import S
# S is numbers?


def generateLLVM(astNode):
    return None


class LLVMPrinter(Printer):
    """Convert expressions to LLVM IR"""
    def __init__(self, module, builder, fn, *args, **kwargs):
        self.func_arg_map = kwargs.pop("func_arg_map", {})
        super(LLVMPrinter, self).__init__(*args, **kwargs)
        self.fp_type = ir.DoubleType()
        #self.integer = ir.IntType(64)
        self.module = module
        self.builder = builder
        self.fn = fn
        self.ext_fn = {}  # keep track of wrappers to external functions
        self.tmp_var = {}

    def _add_tmp_var(self, name, value):
        self.tmp_var[name] = value

    def _print_Number(self, n, **kwargs):
        return ir.Constant(self.fp_type, float(n))

    def _print_Integer(self, expr):
        return ir.Constant(self.fp_type, float(expr.p))

    def _print_Symbol(self, s):
        val = self.tmp_var.get(s)
        if not val:
            # look up parameter with name s
            val = self.func_arg_map.get(s)
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
        for node in nodes[1:]:
            e = self.builder.fmul(e, node)
        return e

    def _print_Add(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fadd(e, node)
        return e

        # TODO - assumes all called functions take one double precision argument.
        #        Should have a list of math library functions to validate this.

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
        raise TypeError("Unsupported type for LLVM JIT conversion: %s"
                        % type(expr))


class Eval(object):
    def __init__(self):
        llvm.initialize()
        llvm.initialize_all_targets()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self.target = llvm.Target.from_default_triple()

    def compile(self, module):
        logger.debug('=============Preparse')
        logger.debug(str(module))
        llvmmod = llvm.parse_assembly(str(module))
        llvmmod.verify()
        logger.debug('=============Function in IR')
        logger.debug(str(llvmmod))
        # TODO cpu, features, opt
        cpu = llvm.get_host_cpu_name()
        features = llvm.get_host_cpu_features()
        logger.debug('=======Things')
        logger.debug(cpu)
        logger.debug(features.flatten())
        target_machine = self.target.create_target_machine(cpu=cpu, features=features.flatten(), opt=2)

        logger.debug('Machine = ' + str(target_machine.target_data))

        with open('gen.ll', 'w') as f:
            f.write(str(llvmmod))
        optimize = True
        if optimize:
            pmb = llvm.create_pass_manager_builder()
            pmb.opt_level = 2
            pmb.disable_unit_at_a_time = False
            pmb.loop_vectorize = True
            pmb.slp_vectorize = True
            # TODO possible to pass for functions
            pm = llvm.create_module_pass_manager()
            pm.add_instruction_combining_pass()
            pm.add_function_attrs_pass()
            pm.add_constant_merge_pass()
            pm.add_licm_pass()
            pmb.populate(pm)
            pm.run(llvmmod)
            logger.debug("==========Opt")
            logger.debug(str(llvmmod))
            with open('gen_opt.ll', 'w') as f:
                f.write(str(llvmmod))

        with llvm.create_mcjit_compiler(llvmmod, target_machine) as ee:
            ee.finalize_object()

            logger.debug('==========Machine code')
            logger.debug(target_machine.emit_assembly(llvmmod))
            with open('gen.S', 'w') as f:
                f.write(target_machine.emit_assembly(llvmmod))
            with open('gen.o', 'wb') as f:
                f.write(target_machine.emit_object(llvmmod))

            # fptr = CFUNCTYPE(c_double, c_double, c_double)(ee.get_function_address('add2'))
            # result = fptr(2, 3)
            # print(result)
            return 0


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)

