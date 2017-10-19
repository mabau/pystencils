import llvmlite.ir as ir
import llvmlite.binding as llvm
import numpy as np
import ctypes as ct
import subprocess
import shutil

from ..data_types import toCtypes, createType, ctypes_from_llvm
from .llvm import generateLLVM
from ..cpu.cpujit import buildCTypeArgumentList, makePythonFunctionIncompleteParams


def generate_and_jit(ast):
    gen = generateLLVM(ast)
    if isinstance(gen, ir.Module):
        return compileLLVM(gen)
    else:
        return compileLLVM(gen.module)


def makePythonFunction(ast, argumentDict={}, func=None):
    if func is None:
        jit = generate_and_jit(ast)
        func = jit.get_function_ptr(ast.functionName)
    try:
        args = buildCTypeArgumentList(ast.parameters, argumentDict)
    except KeyError:
        # not all parameters specified yet
        return makePythonFunctionIncompleteParams(ast, argumentDict, func)
    return lambda: func(*args)


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
        self._llvmmod = llvm.parse_assembly("")
        self.target = llvm.Target.from_default_triple()
        self.cpu = llvm.get_host_cpu_name()
        self.cpu_features = llvm.get_host_cpu_features()
        self.target_machine = self.target.create_target_machine(cpu=self.cpu, features=self.cpu_features.flatten(), opt=2)
        llvm.check_jit_execution()
        self.ee = llvm.create_mcjit_compiler(self.llvmmod, self.target_machine)
        self.ee.finalize_object()
        self.fptr = None

    @property
    def llvmmod(self):
        return self._llvmmod

    @llvmmod.setter
    def llvmmod(self, mod):
        self.ee.remove_module(self.llvmmod)
        self.ee.add_module(mod)
        self.ee.finalize_object()
        self.compile()
        self._llvmmod = mod

    def parse(self, module):
        self.module = module
        llvmmod = llvm.parse_assembly(str(module))
        llvmmod.verify()
        llvmmod.triple = self.target.triple
        llvmmod.name = 'module'
        self.llvmmod = llvmmod

    def write_ll(self, file):
        with open(file, 'w') as f:
            f.write(str(self.llvmmod))

    def write_assembly(self, file):
        with open(file, 'w') as f:
            f.write(self.target_machine.emit_assembly(self.llvmmod))

    def write_object_file(self, file):
        with open(file, 'wb') as f:
            f.write(self.target_machine.emit_object(self.llvmmod))

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

    def optimize_polly(self, opt):
        if shutil.which(opt) is None:
            print('Path to the executable is wrong')
            return
        canonicalize = subprocess.Popen([opt, '-polly-canonicalize'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        analyze = subprocess.Popen(
            [opt, '-polly-codegen', '-polly-vectorizer=polly', '-polly-parallel', '-polly-process-unprofitable', '-f'],
            stdin=canonicalize.stdout, stdout=subprocess.PIPE)

        canonicalize.communicate(input=self.llvmmod.as_bitcode())

        optimize = subprocess.Popen([opt, '-O3', '-f'], stdin=analyze.stdout, stdout=subprocess.PIPE)
        opts, _ = optimize.communicate()
        llvmmod = llvm.parse_bitcode(opts)
        llvmmod.verify()
        self.llvmmod = llvmmod

    def compile(self):
        fptr = {}
        for function in self.module.functions:
            if not function.is_declaration:
                return_type = None
                if function.ftype.return_type != ir.VoidType():
                    return_type = toCtypes(createType(str(function.ftype.return_type)))
                args = [toCtypes(createType(str(arg))) for arg in function.ftype.args]
                function_address = self.ee.get_function_address(function.name)
                fptr[function.name] = ct.CFUNCTYPE(return_type, *args)(function_address)
        self.fptr = fptr

    def __call__(self, function, *args, **kwargs):
        target_function = next(f for f in self.module.functions if f.name == function)
        arg_types = [ctypes_from_llvm(arg.type) for arg in target_function.args]

        transformed_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                transformed_args.append(arg.ctypes.data_as(arg_types[i]))
            else:
                transformed_args.append(arg)

        self.fptr[function](*transformed_args)

    def print_functions(self):
        for f in self.module.functions:
            print(f.ftype.return_type, f.name, f.args)

    def get_function_ptr(self, name):
        fptr = self.fptr[name]
        fptr.jit = self
        return fptr


