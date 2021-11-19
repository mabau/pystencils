from typing import List, Union

import sympy as sp
import numpy as np

import pystencils.astnodes as ast
from pystencils.assignment import Assignment
from pystencils.enums import Target, Backend
from pystencils.astnodes import Block, KernelFunction, LoopOverCoordinate, SympyAssignment
from pystencils.cpu.cpujit import make_python_function
from pystencils.data_types import StructType, TypedSymbol, create_type
from pystencils.field import Field, FieldType
from pystencils.transformations import (
    add_types, filtered_tree_iteration, get_base_buffer_index, get_optimal_loop_ordering, make_loop_over_domain,
    move_constants_before_loop, parse_base_pointer_info, resolve_buffer_accesses,
    resolve_field_accesses, split_inner_loop)

AssignmentOrAstNodeList = List[Union[Assignment, ast.Node]]


def create_kernel(assignments: AssignmentOrAstNodeList, function_name: str = "kernel", type_info='double',
                  split_groups=(), iteration_slice=None, ghost_layers=None,
                  skip_independence_check=False, allow_double_writes=False) -> KernelFunction:
    """Creates an abstract syntax tree for a kernel function, by taking a list of update rules.

    Loops are created according to the field accesses in the equations.

    Args:
        assignments: list of sympy equations, containing accesses to :class:`pystencils.field.Field`.
        Defining the update rules of the kernel
        function_name: name of the generated function - only important if generated code is written out
        type_info: a map from symbol name to a C type specifier. If not specified all symbols are assumed to
                   be of type 'double' except symbols which occur on the left hand side of equations where the
                   right hand side is a sympy Boolean which are assumed to be 'bool' .
        split_groups: Specification on how to split up inner loop into multiple loops. For details see
                      transformation :func:`pystencils.transformation.split_inner_loop`
        iteration_slice: if not None, iteration is done only over this slice of the field
        ghost_layers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                      that should be excluded from the iteration.
                     if None, the number of ghost layers is determined automatically and assumed to be equal for a
                     all dimensions
        skip_independence_check: don't check that loop iterations are independent. This is needed e.g. for
                                 periodicity kernel, that access the field outside the iteration bounds. Use with care!
        allow_double_writes: If True, don't check if every field is only written at a single location. This is required
                             for example for kernels that are compiled with loop step sizes > 1, that handle multiple 
                             cells at once. Use with care!

    Returns:
        AST node representing a function, that can be printed as C or CUDA code
    """
    def type_symbol(term):
        if isinstance(term, Field.Access) or isinstance(term, TypedSymbol):
            return term
        elif isinstance(term, sp.Symbol):
            if isinstance(type_info, str) or not hasattr(type_info, '__getitem__'):
                return TypedSymbol(term.name, create_type(type_info))
            else:
                return TypedSymbol(term.name, type_info[term.name])
        else:
            raise ValueError("Term has to be field access or symbol")

    fields_read, fields_written, assignments = add_types(
        assignments, type_info, not skip_independence_check, check_double_write_condition=not allow_double_writes)
    all_fields = fields_read.union(fields_written)
    read_only_fields = set([f.name for f in fields_read - fields_written])

    buffers = set([f for f in all_fields if FieldType.is_buffer(f)])
    fields_without_buffers = all_fields - buffers

    body = ast.Block(assignments)
    loop_order = get_optimal_loop_ordering(fields_without_buffers)
    loop_node, ghost_layer_info = make_loop_over_domain(body, iteration_slice=iteration_slice,
                                                        ghost_layers=ghost_layers, loop_order=loop_order)
    ast_node = KernelFunction(loop_node, Target.CPU, Backend.C, compile_function=make_python_function,
                              ghost_layers=ghost_layer_info, function_name=function_name, assignments=assignments)

    if split_groups:
        typed_split_groups = [[type_symbol(s) for s in split_group] for split_group in split_groups]
        split_inner_loop(ast_node, typed_split_groups)

    base_pointer_spec = [['spatialInner0'], ['spatialInner1']] if len(loop_order) >= 2 else [['spatialInner0']]
    base_pointer_info = {field.name: parse_base_pointer_info(base_pointer_spec, loop_order,
                                                             field.spatial_dimensions, field.index_dimensions)
                         for field in fields_without_buffers}

    buffer_base_pointer_info = {field.name: parse_base_pointer_info([['spatialInner0']], [0],
                                                                    field.spatial_dimensions, field.index_dimensions)
                                for field in buffers}
    base_pointer_info.update(buffer_base_pointer_info)

    if any(FieldType.is_buffer(f) for f in all_fields):
        resolve_buffer_accesses(ast_node, get_base_buffer_index(ast_node), read_only_fields)
    resolve_field_accesses(ast_node, read_only_fields, field_to_base_pointer_info=base_pointer_info)
    move_constants_before_loop(ast_node)
    return ast_node


def create_indexed_kernel(assignments: AssignmentOrAstNodeList, index_fields, function_name="kernel",
                          type_info=None, coordinate_names=('x', 'y', 'z')) -> KernelFunction:
    """
    Similar to :func:`create_kernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separate index_field, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinate_names' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    Args:
        assignments: list of assignments
        index_fields: list of index fields, i.e. 1D fields with struct data type
        type_info: see documentation of :func:`create_kernel`
        function_name: see documentation of :func:`create_kernel`
        coordinate_names: name of the coordinate fields in the struct data type
    """
    fields_read, fields_written, assignments = add_types(assignments, type_info, check_independence_condition=False)
    all_fields = fields_read.union(fields_written)

    for index_field in index_fields:
        index_field.field_type = FieldType.INDEXED
        assert FieldType.is_indexed(index_field)
        assert index_field.spatial_dimensions == 1, "Index fields have to be 1D"

    non_index_fields = [f for f in all_fields if f not in index_fields]
    spatial_coordinates = {f.spatial_dimensions for f in non_index_fields}
    assert len(spatial_coordinates) == 1, "Non-index fields do not have the same number of spatial coordinates"
    spatial_coordinates = list(spatial_coordinates)[0]

    def get_coordinate_symbol_assignment(name):
        for idx_field in index_fields:
            assert isinstance(idx_field.dtype, StructType), "Index fields have to have a struct data type"
            data_type = idx_field.dtype
            if data_type.has_element(name):
                rhs = idx_field[0](name)
                lhs = TypedSymbol(name, np.int64)
                return SympyAssignment(lhs, rhs)
        raise ValueError(f"Index {name} not found in any of the passed index fields")

    coordinate_symbol_assignments = [get_coordinate_symbol_assignment(n)
                                     for n in coordinate_names[:spatial_coordinates]]
    coordinate_typed_symbols = [eq.lhs for eq in coordinate_symbol_assignments]
    assignments = coordinate_symbol_assignments + assignments

    # make 1D loop over index fields
    loop_body = Block([])
    loop_node = LoopOverCoordinate(loop_body, coordinate_to_loop_over=0, start=0, stop=index_fields[0].shape[0])

    for assignment in assignments:
        loop_body.append(assignment)

    function_body = Block([loop_node])
    ast_node = KernelFunction(function_body, Target.CPU, Backend.C, make_python_function,
                              ghost_layers=None, function_name=function_name, assignments=assignments)

    fixed_coordinate_mapping = {f.name: coordinate_typed_symbols for f in non_index_fields}

    read_only_fields = set([f.name for f in fields_read - fields_written])
    resolve_field_accesses(ast_node, read_only_fields, field_to_fixed_coordinates=fixed_coordinate_mapping)
    move_constants_before_loop(ast_node)
    return ast_node


def add_openmp(ast_node, schedule="static", num_threads=True, collapse=None, assume_single_outer_loop=True):
    """Parallelize the outer loop with OpenMP.

    Args:
        ast_node: abstract syntax tree created e.g. by :func:`create_kernel`
        schedule: OpenMP scheduling policy e.g. 'static' or 'dynamic'
        num_threads: explicitly specify number of threads
        collapse: number of nested loops to include in parallel region (see OpenMP collapse)
        assume_single_outer_loop: if True an exception is raised if multiple outer loops are detected for all but
                                  optimized staggered kernels the single-outer-loop assumption should be true
    """
    if not num_threads:
        return

    assert type(ast_node) is ast.KernelFunction
    body = ast_node.body
    threads_clause = "" if num_threads and isinstance(num_threads, bool) else f" num_threads({num_threads})"
    wrapper_block = ast.PragmaBlock('#pragma omp parallel' + threads_clause, body.take_child_nodes())
    body.append(wrapper_block)

    outer_loops = [l for l in filtered_tree_iteration(body, LoopOverCoordinate, stop_type=SympyAssignment)
                   if l.is_outermost_loop]
    assert outer_loops, "No outer loop found"
    if assume_single_outer_loop and len(outer_loops) > 1:
        raise ValueError("More than one outer loop found, only one outer loop expected")

    for loop_to_parallelize in outer_loops:
        try:
            loop_range = int(loop_to_parallelize.stop - loop_to_parallelize.start)
        except TypeError:
            loop_range = None

        if loop_range is not None and loop_range < num_threads and not collapse:
            contained_loops = [l for l in loop_to_parallelize.body.args if isinstance(l, LoopOverCoordinate)]
            if len(contained_loops) == 1:
                contained_loop = contained_loops[0]
                try:
                    contained_loop_range = int(contained_loop.stop - contained_loop.start)
                    if contained_loop_range > loop_range:
                        loop_to_parallelize = contained_loop
                except TypeError:
                    pass

        prefix = f"#pragma omp for schedule({schedule})"
        if collapse:
            prefix += f" collapse({collapse})"
        loop_to_parallelize.prefix_lines.append(prefix)
