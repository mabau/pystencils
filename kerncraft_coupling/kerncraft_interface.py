from tempfile import TemporaryDirectory

import sympy as sp
import os
from collections import defaultdict
import subprocess
import kerncraft
import kerncraft.kernel
from kerncraft.machinemodel import MachineModel
from kerncraft.models import ECM, Benchmark
from kerncraft.iaca import iaca_analyse_instrumented_binary, iaca_instrumentation
from pystencils.kerncraft_coupling.generate_benchmark import generateBenchmark
from pystencils.astnodes import LoopOverCoordinate, SympyAssignment, ResolvedFieldAccess
from pystencils.field import getLayoutFromStrides
from pystencils.sympyextensions import count_operations_in_ast
from pystencils.utils import DotDict


class PyStencilsKerncraftKernel(kerncraft.kernel.Kernel):
    """
    Implementation of kerncraft's kernel interface for pystencils CPU kernels.
    Analyses a list of equations assuming they will be executed on a CPU
    """
    LIKWID_BASE = '/usr/local/likwid'

    def __init__(self, ast, machine=None):
        super(PyStencilsKerncraftKernel, self).__init__(machine)

        self.ast = ast
        self.temporaryDir = TemporaryDirectory()

        # Loops
        innerLoops = [l for l in ast.atoms(LoopOverCoordinate) if l.isInnermostLoop]
        if len(innerLoops) == 0:
            raise ValueError("No loop found in pystencils AST")
        elif len(innerLoops) > 1:
            raise ValueError("pystencils AST contains multiple inner loops - only one can be analyzed")
        else:
            innerLoop = innerLoops[0]

        self._loop_stack = []
        curNode = innerLoop
        while curNode is not None:
            if isinstance(curNode, LoopOverCoordinate):
                loopCounterSym = curNode.loopCounterSymbol
                loopInfo = (loopCounterSym.name, curNode.start, curNode.stop, curNode.step)
                self._loop_stack.append(loopInfo)
            curNode = curNode.parent
        self._loop_stack = list(reversed(self._loop_stack))

        # Data sources & destinations
        self.sources = defaultdict(list)
        self.destinations = defaultdict(list)

        reads, writes = searchResolvedFieldAccessesInAst(innerLoop)
        for accesses, targetDict in [(reads, self.sources), (writes, self.destinations)]:
            for fa in accesses:
                coord = [sp.Symbol(LoopOverCoordinate.getLoopCounterName(i), positive=True, integer=True) + off
                         for i, off in enumerate(fa.offsets)]
                coord += list(fa.idxCoordinateValues)
                layout = getLayoutFromStrides(fa.field.strides)
                permutedCoord = [coord[i] for i in layout]
                targetDict[fa.field.name].append(permutedCoord)

        # Variables (arrays)
        fieldsAccessed = ast.fieldsAccessed
        for field in fieldsAccessed:
            layout = getLayoutFromStrides(field.strides)
            permutedShape = list(field.shape[i] for i in layout)
            self.set_variable(field.name, str(field.dtype), tuple(permutedShape))

        for param in ast.parameters:
            if not param.isFieldArgument:
                self.set_variable(param.name, str(param.dtype), None)
                self.sources[param.name] = [None]

        # data type
        self.datatype = list(self.variables.values())[0][0]

        # flops
        operationCount = count_operations_in_ast(innerLoop)
        self._flops = {
            '+': operationCount['adds'],
            '*': operationCount['muls'],
            '/': operationCount['divs'],
        }
        for k in [k for k, v in self._flops.items() if v == 0]:
            del self._flops[k]
        self.check()

    def iaca_analysis(self, micro_architecture, asm_block='auto',
                      pointer_increment='auto_with_manual_fallback', verbose=False):
        compiler, compiler_args = self._machine.get_compiler()
        if '-std=c99' not in compiler_args:
            compiler_args += ['-std=c99']
        headerPath = kerncraft.get_header_path()
    
        compilerCmd = [compiler] + compiler_args + ['-I' + headerPath]
    
        srcFile = os.path.join(self.temporaryDir.name, "source.c")
        asmFile = os.path.join(self.temporaryDir.name, "source.s")
        iacaAsmFile = os.path.join(self.temporaryDir.name, "source.iaca.s")
        dummySrcFile = os.path.join(headerPath, "dummy.c")
        dummyAsmFile = os.path.join(self.temporaryDir.name, "dummy.s")
        binaryFile = os.path.join(self.temporaryDir.name, "binary")

        # write source code to file
        with open(srcFile, 'w') as f:
            f.write(generateBenchmark(self.ast, likwid=False))

        # compile to asm files
        subprocess.check_output(compilerCmd + [srcFile,      '-S', '-o', asmFile])
        subprocess.check_output(compilerCmd + [dummySrcFile, '-S', '-o', dummyAsmFile])

        with open(asmFile) as read, open(iacaAsmFile, 'w') as write:
            instrumentedAsmBlock = iaca_instrumentation(read, write)

        # assemble asm files to executable
        subprocess.check_output(compilerCmd + [iacaAsmFile, dummyAsmFile, '-o', binaryFile])

        result = iaca_analyse_instrumented_binary(binaryFile, micro_architecture)
    
        return result, instrumentedAsmBlock

    def build(self, lflags=None, verbose=False):
        compiler, compiler_args = self._machine.get_compiler()
        if '-std=c99' not in compiler_args:
            compiler_args.append('-std=c99')
        headerPath = kerncraft.get_header_path()

        cmd = [compiler] + compiler_args + [
            '-I' + os.path.join(self.LIKWID_BASE, 'include'),
            '-L' + os.path.join(self.LIKWID_BASE, 'lib'),
            '-I' + headerPath,
            '-Wl,-rpath=' + os.path.join(self.LIKWID_BASE, 'lib'),
        ]

        dummySrcFile = os.path.join(headerPath, 'dummy.c')
        srcFile = os.path.join(self.temporaryDir.name, "source_likwid.c")
        binFile = os.path.join(self.temporaryDir.name, "benchmark")

        with open(srcFile, 'w') as f:
            f.write(generateBenchmark(self.ast, likwid=True))

        subprocess.check_output(cmd + [srcFile, dummySrcFile, '-pthread', '-llikwid', '-o', binFile])
        return binFile


class KerncraftParameters(DotDict):
    def __init__(self):
        self['asm_block'] = 'auto'
        self['asm_increment'] = 0
        self['cores'] = 1
        self['cache_predictor'] = 'SIM'
        self['verbose'] = 0
        self['pointer_increment'] = 'auto'
        self['iterations'] = 10


# ------------------------------------------- Helper functions ---------------------------------------------------------


def searchResolvedFieldAccessesInAst(ast):
    def visit(node, reads, writes):
        if not isinstance(node, SympyAssignment):
            for a in node.args:
                visit(a, reads, writes)
            return

        for expr, accesses in [(node.lhs, writes), (node.rhs, reads)]:
            accesses.update(expr.atoms(ResolvedFieldAccess))

    readAccesses = set()
    writeAccesses = set()
    visit(ast, readAccesses, writeAccesses)
    return readAccesses, writeAccesses