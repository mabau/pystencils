import llvmlite.binding as llvm
import logging.config

logger = logging.getLogger(__name__)


def compileLLVM(module):
    return Eval().compile(module)


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
