import ctypes as ct
import subprocess
from functools import partial
from itertools import chain
from os.path import exists, join

import llvmlite.binding as llvm
import llvmlite.ir as ir
import numpy as np

from pystencils.data_types import create_composite_type_from_string
from pystencils.field import FieldType

from ..data_types import StructType, ctypes_from_llvm, to_ctypes
from .llvm import generate_llvm


def build_ctypes_argument_list(parameter_specification, argument_dict):
    argument_dict = {k: v for k, v in argument_dict.items()}
    ct_arguments = []
    array_shapes = set()
    index_arr_shapes = set()

    for param in parameter_specification:
        if param.is_field_parameter:
            try:
                field_arr = argument_dict[param.field_name]
            except KeyError:
                raise KeyError("Missing field parameter for kernel call " + param.field_name)

            symbolic_field = param.fields[0]
            if param.is_field_pointer:
                ct_arguments.append(field_arr.ctypes.data_as(to_ctypes(param.symbol.dtype)))
                if symbolic_field.has_fixed_shape:
                    symbolic_field_shape = tuple(int(i) for i in symbolic_field.shape)
                    if isinstance(symbolic_field.dtype, StructType):
                        symbolic_field_shape = symbolic_field_shape[:-1]
                    if symbolic_field_shape != field_arr.shape:
                        raise ValueError("Passed array '%s' has shape %s which does not match expected shape %s" %
                                         (param.field_name, str(field_arr.shape), str(symbolic_field.shape)))
                if symbolic_field.has_fixed_shape:
                    symbolic_field_strides = tuple(int(i) * field_arr.itemsize for i in symbolic_field.strides)
                    if isinstance(symbolic_field.dtype, StructType):
                        symbolic_field_strides = symbolic_field_strides[:-1]
                    if symbolic_field_strides != field_arr.strides:
                        raise ValueError("Passed array '%s' has strides %s which does not match expected strides %s" %
                                         (param.field_name, str(field_arr.strides), str(symbolic_field_strides)))

                if FieldType.is_indexed(symbolic_field):
                    index_arr_shapes.add(field_arr.shape[:symbolic_field.spatial_dimensions])
                elif FieldType.is_generic(symbolic_field):
                    array_shapes.add(field_arr.shape[:symbolic_field.spatial_dimensions])

            elif param.is_field_shape:
                data_type = to_ctypes(param.symbol.dtype)
                ct_arguments.append(data_type(field_arr.shape[param.symbol.coordinate]))
            elif param.is_field_stride:
                data_type = to_ctypes(param.symbol.dtype)
                assert field_arr.strides[param.symbol.coordinate] % field_arr.itemsize == 0
                item_stride = field_arr.strides[param.symbol.coordinate] // field_arr.itemsize
                ct_arguments.append(data_type(item_stride))
            else:
                assert False
        else:
            try:
                value = argument_dict[param.symbol.name]
            except KeyError:
                raise KeyError("Missing parameter for kernel call " + param.symbol.name)
            expected_type = to_ctypes(param.symbol.dtype)
            ct_arguments.append(expected_type(value))

    if len(array_shapes) > 1:
        raise ValueError("All passed arrays have to have the same size " + str(array_shapes))
    if len(index_arr_shapes) > 1:
        raise ValueError("All passed index arrays have to have the same size " + str(array_shapes))

    return ct_arguments


def make_python_function_incomplete_params(kernel_function_node, argument_dict, func):
    parameters = kernel_function_node.get_parameters()

    cache = {}
    cache_values = []

    def wrapper(**kwargs):
        key = hash(tuple((k, v.ctypes.data, v.strides, v.shape) if isinstance(v, np.ndarray) else (k, id(v))
                         for k, v in kwargs.items()))
        try:
            args = cache[key]
            func(*args)
        except KeyError:
            full_arguments = argument_dict.copy()
            full_arguments.update(kwargs)
            args = build_ctypes_argument_list(parameters, full_arguments)
            cache[key] = args
            cache_values.append(kwargs)  # keep objects alive such that ids remain unique
            func(*args)
    wrapper.ast = kernel_function_node
    wrapper.parameters = kernel_function_node.get_parameters()
    return wrapper


def generate_and_jit(ast):
    target = 'gpu' if ast._backend == 'llvm_gpu' else 'cpu'
    gen = generate_llvm(ast, target=target)
    if isinstance(gen, ir.Module):
        return compile_llvm(gen, target, ast)
    else:
        return compile_llvm(gen.module, target, ast)


def make_python_function(ast, argument_dict={}, func=None):
    if func is None:
        jit = generate_and_jit(ast)
        func = jit.get_function_ptr(ast.function_name)
    try:
        args = build_ctypes_argument_list(ast.get_parameters(), argument_dict)
    except KeyError:
        # not all parameters specified yet
        return make_python_function_incomplete_params(ast, argument_dict, func)
    return lambda: func(*args)


def compile_llvm(module, target='cpu', ast=None):
    jit = CudaJit(ast) if target == "gpu" else Jit()
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
        self.target_machine = self.target.create_target_machine(cpu=self.cpu, features=self.cpu_features.flatten(),
                                                                opt=2)
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

    def compile(self):
        fptr = {}
        for func in self.module.functions:
            if not func.is_declaration:
                return_type = None
                if func.ftype.return_type != ir.VoidType():
                    return_type = to_ctypes(create_composite_type_from_string(str(func.ftype.return_type)))
                args = [ctypes_from_llvm(arg) for arg in func.ftype.args]
                function_address = self.ee.get_function_address(func.name)
                fptr[func.name] = ct.CFUNCTYPE(return_type, *args)(function_address)
        self.fptr = fptr

    def __call__(self, func, *args, **kwargs):
        target_function = next(f for f in self.module.functions if f.name == func)
        arg_types = [ctypes_from_llvm(arg.type) for arg in target_function.args]

        transformed_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                transformed_args.append(arg.ctypes.data_as(arg_types[i]))
            else:
                transformed_args.append(arg)

        self.fptr[func](*transformed_args)

    def print_functions(self):
        for f in self.module.functions:
            print(f.ftype.return_type, f.name, f.args)

    def get_function_ptr(self, name):
        fptr = self.fptr[name]
        fptr.jit = self
        return fptr


# Following code more or less from numba
class CudaJit(Jit):

    CUDA_TRIPLE = {32: 'nvptx-nvidia-cuda',
                   64: 'nvptx64-nvidia-cuda'}
    MACHINE_BITS = tuple.__itemsize__ * 8
    data_layout = {
        32: ('e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-'
             'f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64'),
        64: ('e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-'
             'f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64')}

    default_data_layout = data_layout[MACHINE_BITS]

    def __init__(self, ast):
        # super().__init__()

        # self.target = llvm.Target.from_triple(self.CUDA_TRIPLE[self.MACHINE_BITS])
        self._data_layout = self.default_data_layout[self.MACHINE_BITS]
        # self._target_data = llvm.create_target_data(self._data_layout)
        self.indexing = ast.indexing

    def optimize(self):
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 2
        pmb.disable_unit_at_a_time = False
        pmb.loop_vectorize = False
        pmb.slp_vectorize = False
        # TODO possible to pass for functions
        pm = llvm.create_module_pass_manager()
        pm.add_instruction_combining_pass()
        pm.add_function_attrs_pass()
        pm.add_constant_merge_pass()
        pm.add_licm_pass()
        pmb.populate(pm)
        pm.run(self.llvmmod)
        pm.run(self.llvmmod)

    def write_ll(self, file):
        with open(file, 'w') as f:
            f.write(str(self.llvmmod))

    def parse(self, module):

        llvmmod = module
        llvmmod.triple = self.CUDA_TRIPLE[self.MACHINE_BITS]
        llvmmod.data_layout = self.default_data_layout
        llvmmod.verify()
        llvmmod.name = 'module'

        self._llvmmod = llvm.parse_assembly(str(llvmmod))

    def compile(self):
        from pystencils.cpu.cpujit import get_cache_config, get_compiler_config, get_llc_command
        import hashlib
        compiler_cache = get_cache_config()['object_cache']
        ir_file = join(compiler_cache, hashlib.md5(str(self._llvmmod).encode()).hexdigest() + '.ll')
        ptx_file = ir_file.replace('.ll', '.ptx')
        try:
            from pycuda.driver import Context
            arch = "sm_%d%d" % Context.get_device().compute_capability()
        except Exception:
            arch = "sm_35"

        if not exists(ptx_file):
            self.write_ll(ir_file)
            if 'llc' in get_compiler_config():
                llc_command = get_compiler_config()['llc']
            else:
                llc_command = get_llc_command() or 'llc'

            subprocess.check_call([llc_command, '-mcpu=' + arch, ir_file, '-o', ptx_file])

        # cubin_file = ir_file.replace('.ll', '.cubin')
        # if not exists(cubin_file):
            # subprocess.check_call(['ptxas', '--gpu-name', arch, ptx_file, '-o', cubin_file])
        import pycuda.driver

        cuda_module = pycuda.driver.module_from_file(ptx_file)  # also works: cubin_file
        self.cuda_module = cuda_module

    def __call__(self, func, *args, **kwargs):
        shape = [a.shape for a in chain(args, kwargs.values()) if hasattr(a, 'shape')][0]
        block_and_thread_numbers = self.indexing.call_parameters(shape)
        block_and_thread_numbers['block'] = tuple(int(i) for i in block_and_thread_numbers['block'])
        block_and_thread_numbers['grid'] = tuple(int(i) for i in block_and_thread_numbers['grid'])
        self.cuda_module.get_function(func)(*args, **kwargs, **block_and_thread_numbers)

    def get_function_ptr(self, name):
        return partial(self._call__, name)
