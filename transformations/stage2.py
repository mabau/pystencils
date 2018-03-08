from operator import attrgetter

import sympy as sp

from pystencils.data_types import TypedSymbol, createType, PointerType, StructType, getBaseType, getTypeOfExpression, collateTypes, castFunc, pointerArithmeticFunc
import pystencils.astnodes as ast


def insertCasts(node):
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
            if dataType.numpyDtype != target.numpyDtype:  # ignoring const
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
            if dataType.func is PointerType:
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
    # TODO indexed, LoopOverCoordinate
    if node.func in (sp.Add, sp.Mul, sp.Or, sp.And, sp.Pow, sp.Eq, sp.Ne, sp.Lt, sp.Le, sp.Gt, sp.Ge):
        # TODO optimize pow, don't cast integer on double
        types = [getTypeOfExpression(arg) for arg in args]
        assert len(types) > 0
        target = collateTypes(types)
        zipped = list(zip(args, types))
        if target.func is PointerType:
            assert node.func is sp.Add
            return pointerArithmetic(zipped)
        else:
            return node.func(*cast(zipped, target))
    elif node.func is ast.SympyAssignment:
        lhs = args[0]
        rhs = args[1]
        target = getTypeOfExpression(lhs)
        if target.func is PointerType:
            return node.func(*args)  # TODO fix, not complete
        else:
            return node.func(lhs, *cast([(rhs, getTypeOfExpression(rhs))], target))
    elif node.func is ast.ResolvedFieldAccess:
        return node
    elif node.func is ast.Block:
        for oldArg, newArg in zip(node.args, args):
            node.replace(oldArg, newArg)
        return node
    elif node.func is ast.LoopOverCoordinate:
        for oldArg, newArg in zip(node.args, args):
            node.replace(oldArg, newArg)
        return node
    elif node.func is sp.Piecewise:
        exprs = [expr for (expr, _) in args]
        types = [getTypeOfExpression(expr) for expr in exprs]
        target = collateTypes(types)
        zipped = list(zip(exprs, types))
        casted_exprs = cast(zipped, target)
        args = [arg.func(*[expr, arg.cond]) for (arg, expr) in zip(args, casted_exprs)]

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
#            node.replace(arg, arg.target)
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
