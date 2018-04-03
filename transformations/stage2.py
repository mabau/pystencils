import sympy as sp
from pystencils.data_types import PointerType, get_type_of_expression, collate_types, cast_func, pointer_arithmetic_func
import pystencils.astnodes as ast


def insert_casts(node):
    """
    Checks the types and inserts casts and pointer arithmetic where necessary
    :param node: the head node of the ast
    :return: modified ast
    """
    def cast(zipped_args_types, target_dtype):
        """
        Adds casts to the arguments if their type differs from the target type
        :param zipped_args_types: a zipped list of args and types
        :param target_dtype: The target data type
        :return: args with possible casts
        """
        casted_args = []
        for argument, dataType in zipped_args_types:
            if dataType.numpy_dtype != target_dtype.numpy_dtype:  # ignoring const
                casted_args.append(cast_func(argument, target_dtype))
            else:
                casted_args.append(argument)
        return casted_args

    def pointer_arithmetic(expr_args):
        """
        Creates a valid pointer arithmetic function
        :param expr_args: Arguments of the add expression
        :return: pointer_arithmetic_func
        """
        pointer = None
        new_args = []
        for arg, dataType in expr_args:
            if dataType.func is PointerType:
                assert pointer is None
                pointer = arg
        for arg, dataType in expr_args:
            if arg != pointer:
                assert dataType.is_int() or dataType.is_uint()
                new_args.append(arg)
        new_args = sp.Add(*new_args) if len(new_args) > 0 else new_args
        return pointer_arithmetic_func(pointer, new_args)

    if isinstance(node, sp.AtomicExpr):
        return node
    args = []
    for arg in node.args:
        args.append(insert_casts(arg))
    # TODO indexed, LoopOverCoordinate
    if node.func in (sp.Add, sp.Mul, sp.Or, sp.And, sp.Pow, sp.Eq, sp.Ne, sp.Lt, sp.Le, sp.Gt, sp.Ge):
        # TODO optimize pow, don't cast integer on double
        types = [get_type_of_expression(arg) for arg in args]
        assert len(types) > 0
        target = collate_types(types)
        zipped = list(zip(args, types))
        if target.func is PointerType:
            assert node.func is sp.Add
            return pointer_arithmetic(zipped)
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
        expressions = [expr for (expr, _) in args]
        types = [get_type_of_expression(expr) for expr in expressions]
        target = collate_types(types)
        zipped = list(zip(expressions, types))
        casted_expressions = cast(zipped, target)
        args = [arg.func(*[expr, arg.cond]) for (arg, expr) in zip(args, casted_expressions)]

    return node.func(*args)
