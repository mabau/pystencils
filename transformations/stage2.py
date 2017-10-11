from operator import attrgetter

import sympy as sp

from pystencils.data_types import TypedSymbol, createType, PointerType, StructType, getBaseType, getTypeOfExpression, collateTypes, castFunc, pointerArithmeticFunc
import pystencils.astnodes as ast


def insertCasts(node): # TODO test casts!!!, edit testcase
    """
    Checks the types and inserts casts and pointer arithmetic where necessary
    :param node: the head node of the ast
    :return: modified ast
    """
    def cast(zippedArgsTypes, target):
        """
        Adds casts to the arguments if their type differs from the target type
        :param zippedArgsTypes: a zipped list of args and types
        :param target: The target data type
        :return: args with possible casts
        """
        casted_args = []
        for arg, dataType in zippedArgsTypes:
            if dataType.numpyDtype != target.numpyDtype: # TODO ignoring const
                casted_args.append(castFunc(arg, target))
            else:
                casted_args.append(arg)
        return casted_args

    def pointerArithmetic(args):
        """
        Creates a valid pointer arithmetic function
        :param args: Arguments of the add expression
        :return: pointerArithmeticFunc
        """
        pointer = None
        newArgs = []
        for arg, dataType in args:
            if dataType.func == PointerType:
                assert pointer is None
                pointer = arg
        for arg, dataType in args:
            if arg != pointer:
                assert dataType.is_int() or dataType.is_uint()
                newArgs.append(arg)
        newArgs = sp.Add(*newArgs) if len(newArgs) > 0 else newArgs
        return pointerArithmeticFunc(pointer, newArgs)

    if isinstance(node, sp.AtomicExpr):
        return node
    args = []
    for arg in node.args:
        args.append(insertCasts(arg))
    # TODO indexed, SympyAssignment, LoopOverCoordinate, Pow
    if node.func in (sp.Add, sp.Mul):
        types = [getTypeOfExpression(arg) for arg in args]
        assert len(types) > 0
        target = collateTypes(types)
        zipped = list(zip(args, types))
        print(zipped)
        if target.func == PointerType:
            assert node.func == sp.Add
            return pointerArithmetic(zipped)
        else:
            return node.func(*cast(zipped, target))
    elif node.func == ast.SympyAssignment:
        # TODO casting of rhs/lhs
        return node.func(*args)
    elif node.func == ast.ResolvedFieldAccess:
        #print("Node:", node, type(node), node.__class__.mro())
        # TODO Everything
        return node
    elif node.func == ast.Block:
        for oldArg, newArg in zip(node.args, args):
            node.replace(oldArg, newArg)
        return node
    elif node.func == ast.LoopOverCoordinate:
        for oldArg, newArg in zip(node.args, args):
            node.replace(oldArg, newArg)
        return node

    #print(node.func(*args))
    return node.func(*args)


def insert_casts(node):
    """
    Inserts casts and dtype where needed
    :param node: ast which should be traversed
    :return: node
    """
    def conversion(args):
        target = args[0]
        if isinstance(target.dtype, PointerType):
            # Pointer arithmetic
            for arg in args[1:]:
                # Check validness
                if not arg.dtype.is_int() and not arg.dtype.is_uint():
                    raise ValueError("Impossible pointer arithmetic", target, arg)
            pointer = ast.PointerArithmetic(ast.Add(args[1:]), target)
            return [pointer]

        else:
            for i in range(len(args)):
                if args[i].dtype.numpyDtype != target.dtype.numpyDtype:  # TODO ignoring const -> valid behavior?
                    args[i] = ast.Conversion(args[i], createType(target.dtype), node)
            return args

    for arg in node.args:
        insert_casts(arg)
    if isinstance(node, ast.Indexed):
        # TODO need to do something here?
        pass
    elif isinstance(node, ast.Expr):
        args = sorted((arg for arg in node.args), key=attrgetter('dtype'))
        target = args[0]
        node.args = conversion(args)
        node.dtype = target.dtype
    elif isinstance(node, ast.SympyAssignment):
        if node.lhs.dtype != node.rhs.dtype:
            node.replace(node.rhs, ast.Conversion(node.rhs, node.lhs.dtype))
    elif isinstance(node, ast.LoopOverCoordinate):
        pass
    return node


#def desympy_ast(node):
#    """
#    Remove Sympy Expressions, which have more then one argument.
#    This is necessary for further changes in the tree.
#    :param node: ast which should be traversed. Only node's children will be modified.
#    :return: (modified) node
#    """
#    if node.args is None:
#        return node
#    for i in range(len(node.args)):
#        arg = node.args[i]
#        if isinstance(arg, sp.Add):
#            node.replace(arg, ast.Add(arg.args, node))
#        elif isinstance(arg, sp.Number):
#            node.replace(arg, ast.Number(arg, node))
#        elif isinstance(arg, sp.Mul):
#            node.replace(arg, ast.Mul(arg.args, node))
#        elif isinstance(arg, sp.Pow):
#            node.replace(arg, ast.Pow(arg, node))
#        elif isinstance(arg, sp.tensor.Indexed) or isinstance(arg, sp.tensor.indexed.Indexed):
#            node.replace(arg, ast.Indexed(arg.args, arg.base, node))
#        elif isinstance(arg,  sp.tensor.IndexedBase):
#            node.replace(arg, arg.label)
#        elif isinstance(arg, sp.Function):
#            node.replace(arg, ast.Function(arg.func, arg.args, node))
#        #elif isinstance(arg, sp.containers.Tuple):
#        #
#        else:
#            #print('Not transforming:', type(arg), arg)
#            pass
#    for arg in node.args:
#        desympy_ast(arg)
#    return node
