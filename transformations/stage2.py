import sympy as sp
from pystencils.data_types import PointerType, get_type_of_expression, collate_types, castFunc, pointerArithmeticFunc
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
            if dataType.numpy_dtype != target.numpy_dtype:  # ignoring const
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
        types = [get_type_of_expression(arg) for arg in args]
        assert len(types) > 0
        target = collate_types(types)
        zipped = list(zip(args, types))
        if target.func is PointerType:
            assert node.func is sp.Add
            return pointerArithmetic(zipped)
        else:
            return node.func(*cast(zipped, target))
    elif node.func is ast.SympyAssignment:
        lhs = args[0]
        rhs = args[1]
        target = get_type_of_expression(lhs)
        if target.func is PointerType:
            return node.func(*args)  # TODO fix, not complete
        else:
            return node.func(lhs, *cast([(rhs, get_type_of_expression(rhs))], target))
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
        types = [get_type_of_expression(expr) for expr in exprs]
        target = collate_types(types)
        zipped = list(zip(exprs, types))
        casted_exprs = cast(zipped, target)
        args = [arg.func(*[expr, arg.cond]) for (arg, expr) in zip(args, casted_exprs)]

    return node.func(*args)
