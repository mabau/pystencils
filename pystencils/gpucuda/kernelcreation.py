from typing import Union

import numpy as np

from pystencils.astnodes import Block, KernelFunction, LoopOverCoordinate, SympyAssignment
from pystencils.config import CreateKernelConfig
from pystencils.typing import StructType, TypedSymbol
from pystencils.typing.transformations import add_types
from pystencils.field import Field, FieldType
from pystencils.enums import Target, Backend
from pystencils.gpucuda.cudajit import make_python_function
from pystencils.node_collection import NodeCollection
from pystencils.gpucuda.indexing import indexing_creator_from_params
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.transformations import (
    get_base_buffer_index, get_common_shape, parse_base_pointer_info,
    resolve_buffer_accesses, resolve_field_accesses, unify_shape_symbols)


def create_cuda_kernel(assignments: Union[AssignmentCollection, NodeCollection],
                       config: CreateKernelConfig):

    function_name = config.function_name
    indexing_creator = indexing_creator_from_params(config.gpu_indexing, config.gpu_indexing_params)
    iteration_slice = config.iteration_slice
    ghost_layers = config.ghost_layers

    fields_written = assignments.bound_fields
    fields_read = assignments.rhs_fields
    assignments = assignments.all_assignments

    assignments = add_types(assignments, config)

    all_fields = fields_read.union(fields_written)
    read_only_fields = set([f.name for f in fields_read - fields_written])

    buffers = set([f for f in all_fields if FieldType.is_buffer(f) or FieldType.is_custom(f)])
    fields_without_buffers = all_fields - buffers

    field_accesses = set()
    num_buffer_accesses = 0
    for eq in assignments:
        field_accesses.update(eq.atoms(Field.Access))
        field_accesses = {e for e in field_accesses if not e.is_absolute_access}
        num_buffer_accesses += sum(1 for access in eq.atoms(Field.Access) if FieldType.is_buffer(access.field))

    common_shape = get_common_shape(fields_without_buffers)

    if iteration_slice is None:
        # determine iteration slice from ghost layers
        if ghost_layers is None:
            # determine required number of ghost layers from field access
            required_ghost_layers = max([fa.required_ghost_layers for fa in field_accesses])
            ghost_layers = [(required_ghost_layers, required_ghost_layers)] * len(common_shape)
        iteration_slice = []
        if isinstance(ghost_layers, int):
            for i in range(len(common_shape)):
                iteration_slice.append(slice(ghost_layers, -ghost_layers if ghost_layers > 0 else None))
            ghost_layers = [(ghost_layers, ghost_layers)] * len(common_shape)
        else:
            for i in range(len(common_shape)):
                iteration_slice.append(slice(ghost_layers[i][0],
                                             -ghost_layers[i][1] if ghost_layers[i][1] > 0 else None))

    indexing = indexing_creator(field=list(fields_without_buffers)[0], iteration_slice=iteration_slice)
    coord_mapping = indexing.coordinates

    cell_idx_assignments = [SympyAssignment(LoopOverCoordinate.get_loop_counter_symbol(i), value)
                            for i, value in enumerate(coord_mapping)]
    cell_idx_symbols = [LoopOverCoordinate.get_loop_counter_symbol(i) for i, _ in enumerate(coord_mapping)]
    assignments = cell_idx_assignments + assignments

    block = Block(assignments)

    block = indexing.guard(block, common_shape)
    unify_shape_symbols(block, common_shape=common_shape, fields=fields_without_buffers)

    ast = KernelFunction(block,
                         Target.GPU,
                         Backend.CUDA,
                         make_python_function,
                         ghost_layers,
                         function_name,
                         assignments=assignments)
    ast.global_variables.update(indexing.index_variables)

    base_pointer_spec = [['spatialInner0']]
    base_pointer_info = {f.name: parse_base_pointer_info(base_pointer_spec, [2, 1, 0],
                                                         f.spatial_dimensions, f.index_dimensions)
                         for f in all_fields}

    coord_mapping = {f.name: cell_idx_symbols for f in all_fields}

    if any(FieldType.is_buffer(f) for f in all_fields):
        resolve_buffer_accesses(ast, get_base_buffer_index(ast, indexing.coordinates, common_shape), read_only_fields)

    resolve_field_accesses(ast, read_only_fields, field_to_base_pointer_info=base_pointer_info,
                           field_to_fixed_coordinates=coord_mapping)

    # add the function which determines #blocks and #threads as additional member to KernelFunction node
    # this is used by the jit

    # If loop counter symbols have been explicitly used in the update equations (e.g. for built in periodicity),
    # they are defined here
    undefined_loop_counters = {LoopOverCoordinate.is_loop_counter_symbol(s): s for s in ast.body.undefined_symbols
                               if LoopOverCoordinate.is_loop_counter_symbol(s) is not None}
    for i, loop_counter in undefined_loop_counters.items():
        ast.body.insert_front(SympyAssignment(loop_counter, indexing.coordinates[i]))

    ast.indexing = indexing
    return ast


def created_indexed_cuda_kernel(assignments: Union[AssignmentCollection, NodeCollection],
                                config: CreateKernelConfig):

    index_fields = config.index_fields
    function_name = config.function_name
    coordinate_names = config.coordinate_names
    indexing_creator = indexing_creator_from_params(config.gpu_indexing, config.gpu_indexing_params)

    fields_written = assignments.bound_fields
    fields_read = assignments.rhs_fields
    assignments = assignments.all_assignments

    assignments = add_types(assignments, config)

    all_fields = fields_read.union(fields_written)
    read_only_fields = set([f.name for f in fields_read - fields_written])

    for index_field in index_fields:
        index_field.field_type = FieldType.INDEXED
        assert FieldType.is_indexed(index_field)
        assert index_field.spatial_dimensions == 1, "Index fields have to be 1D"

    non_index_fields = [f for f in all_fields if f not in index_fields]
    spatial_coordinates = {f.spatial_dimensions for f in non_index_fields}
    assert len(spatial_coordinates) == 1, "Non-index fields do not have the same number of spatial coordinates"
    spatial_coordinates = list(spatial_coordinates)[0]

    def get_coordinate_symbol_assignment(name):
        for ind_f in index_fields:
            assert isinstance(ind_f.dtype, StructType), "Index fields have to have a struct data type"
            data_type = ind_f.dtype
            if data_type.has_element(name):
                rhs = ind_f[0](name)
                lhs = TypedSymbol(name, np.int64)
                return SympyAssignment(lhs, rhs)
        raise ValueError(f"Index {name} not found in any of the passed index fields")

    coordinate_symbol_assignments = [get_coordinate_symbol_assignment(n)
                                     for n in coordinate_names[:spatial_coordinates]]
    coordinate_typed_symbols = [eq.lhs for eq in coordinate_symbol_assignments]

    idx_field = list(index_fields)[0]
    indexing = indexing_creator(field=idx_field,
                                iteration_slice=[slice(None, None, None)] * len(idx_field.spatial_shape))

    function_body = Block(coordinate_symbol_assignments + assignments)
    function_body = indexing.guard(function_body, get_common_shape(index_fields))
    ast = KernelFunction(function_body, Target.GPU, Backend.CUDA, make_python_function,
                         None, function_name, assignments=assignments)
    ast.global_variables.update(indexing.index_variables)

    coord_mapping = indexing.coordinates
    base_pointer_spec = [['spatialInner0']]
    base_pointer_info = {f.name: parse_base_pointer_info(base_pointer_spec, [2, 1, 0],
                                                         f.spatial_dimensions, f.index_dimensions)
                         for f in all_fields}

    coord_mapping = {f.name: coord_mapping for f in index_fields}
    coord_mapping.update({f.name: coordinate_typed_symbols for f in non_index_fields})
    resolve_field_accesses(ast, read_only_fields, field_to_fixed_coordinates=coord_mapping,
                           field_to_base_pointer_info=base_pointer_info)

    # add the function which determines #blocks and #threads as additional member to KernelFunction node
    # this is used by the jit
    ast.indexing = indexing
    return ast
