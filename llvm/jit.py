import llvmlite.ir as ir
import llvmlite.binding as llvm
from ..types import toCtypes, createType

import ctypes as ct


def compileLLVM(module):
    jit = Jit()
    jit.parse(module)
    jit.optimize()
    jit.compile()
    return jit


class Jit(object):
    def __init__(self):
        llvm.initialize()
        llvm.initialize_all_targets()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        self.module = None
        self.llvmmod = None
        self.target = llvm.Target.from_default_triple()
        self.cpu = llvm.get_host_cpu_name()
        self.cpu_features = llvm.get_host_cpu_features()
        self.target_machine = self.target.create_target_machine(cpu=self.cpu, features=self.cpu_features.flatten(), opt=2)
        self.ee = None
        self.fptr = None

    def parse(self, module):
        self.module = module
        llvmmod = llvm.parse_assembly(str(module))
        llvmmod.verify()
        self.llvmmod = llvmmod

    def write_ll(self, file):
        with open(file, 'w') as f:
            f.write(str(self.llvmmod))

    def optimize(self):
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
        pm.run(self.llvmmod)

    def compile(self, assembly_file=None, object_file=None):
        ee = llvm.create_mcjit_compiler(self.llvmmod, self.target_machine)
        ee.finalize_object()

        if assembly_file is not None:
            with open(assembly_file, 'w') as f:
                f.write(self.target_machine.emit_assembly(self.llvmmod))
        if object_file is not None:
            with open(object_file, 'wb') as f:
                f.write(self.target_machine.emit_object(self.llvmmod))

        fptr = {}
        for function in self.module.functions:
            if not function.is_declaration:
                return_type = None
                if function.ftype.return_type != ir.VoidType():
                    return_type = toCtypes(createType(str(function.ftype.return_type)))
                args = [toCtypes(createType(str(arg))) for arg in function.ftype.args]
                function_address = ee.get_function_address(function.name)
                fptr[function.name] = ct.CFUNCTYPE(return_type, *args)(function_address)
        self.ee = ee
        self.fptr = fptr

    def __call__(self, function, *args, **kwargs):
        self.fptr[function](*args, **kwargs)
