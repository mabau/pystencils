import sympy as sp
from collections import defaultdict
import kerncraft.kernel
from pystencils.cpu import createKernel
from pystencils.kerncraft.generate_benchmark import generateBenchmark
from pystencils.transformations import typeAllEquations
from pystencils.astnodes import LoopOverCoordinate, SympyAssignment, ResolvedFieldAccess
from pystencils.field import Field, getLayoutFromStrides
from pystencils.sympyextensions import countNumberOfOperations, prod, countNumberOfOperationsInAst
from pystencils.utils import DotDict


class PyStencilsKerncraftKernel(kerncraft.kernel.Kernel):
    """
    Implementation of kerncraft's kernel interface for pystencils CPU kernels.
    Analyses a list of equations assuming they will be executed on a CPU
    """
    def __init__(self, ast):
        super(PyStencilsKerncraftKernel, self).__init__()

        self.ast = ast

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
        self._sources = defaultdict(list)
        self._destinations = defaultdict(list)

        reads, writes = searchResolvedFieldAccessesInAst(innerLoop)
        for accesses, targetDict in [(reads, self._sources), (writes, self._destinations)]:
            for fa in accesses:
                coord = [sp.Symbol(LoopOverCoordinate.getLoopCounterName(i), positive=True) + off
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
            self.set_variable(field.name, str(field.dtype), permutedShape)

        for param in ast.parameters:
            if not param.isFieldArgument:
                self.set_variable(param.name, str(param.dtype), None)
                self._sources[param.name] = [None]

        # data type
        self.datatype = list(self.variables.values())[0][0]

        # flops
        operationCount = countNumberOfOperationsInAst(innerLoop)
        self._flops = {
            '+': operationCount['adds'],
            '*': operationCount['muls'],
            '/': operationCount['divs'],
        }

        self.check()

    def as_code(self, type_='iaca'):
        likwid = type_ == 'likwid'
        generateBenchmark(self.ast, likwid)


class KerncraftParameters(DotDict):
    def __init__(self):
        self['asm_block'] = 'auto'
        self['asm_increment'] = 0
        self['cores'] = 1
        self['cache_predictor'] = 'SIM'
        self['verbose'] = 0



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