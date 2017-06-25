import kerncraft
from kerncraft.kernel import KernelCode
from kernel import KernelDescription
from pystencils.astnodes import LoopOverCoordinate
from pystencils.cpu import createKernel
from pystencils.field import getLayoutFromStrides
from pystencils.sympyextensions import countNumberOfOperations
from pystencils.transformations import typeAllEquations
from pystencils import Field
from collections import defaultdict


class PyStencilsKerncraftKernel(kerncraft.kernel.Kernel):

    def __init__(self, listOfEquations, typeForSymbol=None):
        super(PyStencilsKerncraftKernel, self).__init__()

        pystencilsAst = createKernel(listOfEquations, typeForSymbol=typeForSymbol)
        self.ast = pystencilsAst
        fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
        allFields = fieldsRead.union(fieldsWritten)

        # Loops
        innerLoops = [l for l in pystencilsAst.atoms(LoopOverCoordinate) if l.isInnermostLoop]
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
                loopInfo = (curNode.loopCounterSymbol.name, curNode.start, curNode.stop, curNode.step)
                self._loop_stack.append(loopInfo)
            curNode = curNode.parent
        self._loop_stack = list(reversed(self._loop_stack))

        # Data sources & destinations
        self._sources = defaultdict(list)
        self._destinations = defaultdict(list)
        for eq in listOfEquations:
            for accessesDict, expr in [(self._destinations, eq.lhs), (self._sources, eq.rhs)]:
                for fa in expr.atoms(Field.Access):
                    coord = [sp.Symbol(LoopOverCoordinate.getLoopCounterName(i)) + off for i, off in enumerate(fa.offsets)]
                    coord += list(fa.index)
                    layout = getLayoutFromStrides(fa.field.strides)
                    permutedCoord = [coord[i] for i in layout]
                    accessesDict[fa.field.name].append(permutedCoord)

        # Variables (arrays)
        for field in allFields:
            layout = getLayoutFromStrides(field.strides)
            permutedShape = list(field.shape[i] for i in layout)
            self.set_variable(field.name, str(field.dtype), permutedShape)
        for param in pystencilsAst.parameters:
            if not param.isFieldArgument:
                self.set_variable(param.name, str(param.dtype), None)
                self._sources[param.name] = [None]

        # Datatype
        self.datatype = list(self.variables.values())[0][0]

        # Flops
        operationCount = countNumberOfOperations(listOfEquations)
        self._flops = {
            '+': operationCount['adds'],
            '*': operationCount['muls'],
            '/': operationCount['divs'],
        }

        self.check()

from kerncraft.iaca_marker import find_asm_blocks, userselect_block, select_best_block
from kerncraft.models import ECM, ECMData
from kerncraft.machinemodel import MachineModel
from ruamel import yaml

if __name__ == "__main__":
    from pystencils import Field
    import sympy as sp
    import numpy as np
    from pystencils.cpu import generateC

    arr = np.zeros([80, 40], order='c')
    #arr = np.zeros([40, 80, 3], order='f')
    a = Field.createFromNumpyArray('a', arr, indexDimensions=0)
    b = Field.createFromNumpyArray('b', arr, indexDimensions=0)

    s = sp.Symbol("s")
    rhs = a[0, -1](0) + a[0, 1] + a[-1, 0] + a[1, 0]
    updateRule = sp.Eq(b[0, 0], s*rhs)
    k = PyStencilsKerncraftKernel([updateRule])
    print(generateC(k.ast))
    kernelFile = "2d-5pt.c"
    #k = KernelCode(open("/home/martin/dev/kerncraft/examples/kernels/" + kernelFile).read())]
    descr = yaml.load(open("/home/martin/dev/pystencils/pystencils/kerncraft/2d-5pt.yml").read())
    k = KernelDescription(descr)
    k.print_kernel_info()
    k.print_variables_info()
    offsets = list(k.compile_global_offsets(1000))
    print(offsets)

    machineFilePath = "/home/martin/dev/kerncraft/examples/machine-files/emmy.yaml"
    machine = MachineModel(path_to_yaml=machineFilePath)
    #exit(0)
    from kerncraft.kerncraft import create_parser
    parser = create_parser()
    parserArgs = parser.parse_args(["-m", machineFilePath, "-p", "ECMData", machineFilePath])

    model = ECMData(k, machine, parserArgs)
    model.analyze()
    model.report()
    #blocks = find_asm_blocks(open("/home/martin/dev/kerncraft/2d-5pt.c_compilable.s").readlines())
    #userselect_block(blocks)
    ##select_
    #bestBlock = select_best_block(blocks)
    #print(bestBlock)
