import sympy as sp

from pystencils.astnodes import Block, KernelFunction, LoopOverCoordinate, SympyAssignment
from pystencils.config import CreateKernelConfig
from pystencils.typing import StructType, TypedSymbol
from pystencils.typing.transformations import add_types
from pystencils.field import Field, FieldType
from pystencils.enums import Target, Backend
from pystencils.gpu.gpujit import make_python_function
from pystencils.node_collection import NodeCollection
from pystencils.gpu.indexing import indexing_creator_from_params
from pystencils.slicing import normalize_slice
from pystencils.transformations import (
    get_base_buffer_index, get_common_field, get_common_indexed_element, parse_base_pointer_info,
    resolve_buffer_accesses, resolve_field_accesses, unify_shape_symbols)


def create_cuda_kernel(assignments: NodeCollection, config: CreateKernelConfig):

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

    buffers = set([f for f in all_fields if FieldType.is_buffer(f)])
    fields_without_buffers = all_fields - buffers

    field_accesses = set()
    num_buffer_accesses = 0
    indexed_elements = set()
    for eq in assignments:
        indexed_elements.update(eq.atoms(sp.Indexed))
        field_accesses.update(eq.atoms(Field.Access))
        field_accesses = {e for e in field_accesses if not e.is_absolute_access}
        num_buffer_accesses += sum(1 for access in eq.atoms(Field.Access) if FieldType.is_buffer(access.field))

    # common shape and field to from the iteration space
    common_field = get_common_field(fields_without_buffers)
    common_shape = common_field.spatial_shape

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

        iteration_space = normalize_slice(iteration_slice, common_shape)
    else:
        iteration_space = normalize_slice(iteration_slice, common_shape)
    iteration_space = tuple([s if isinstance(s, slice) else slice(s, s, 1) for s in iteration_space])

    loop_counter_symbols = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(len(iteration_space))]

    if len(indexed_elements) > 0:
        common_indexed_element = get_common_indexed_element(indexed_elements)
        indexing = indexing_creator(iteration_space=(slice(0, common_indexed_element.shape[0], 1), *iteration_space),
                                    data_layout=common_field.layout)
        extended_ctrs = [common_indexed_element.indices[0], *loop_counter_symbols]
        loop_counter_assignments = indexing.get_loop_ctr_assignments(extended_ctrs)
    else:
        indexing = indexing_creator(iteration_space=iteration_space, data_layout=common_field.layout)
        loop_counter_assignments = indexing.get_loop_ctr_assignments(loop_counter_symbols)
    assignments = loop_counter_assignments + assignments
    block = indexing.guard(Block(assignments), common_shape)

    unify_shape_symbols(block, common_shape=common_shape, fields=fields_without_buffers)

    ast = KernelFunction(block,
                         Target.GPU,
                         Backend.CUDA,
                         make_python_function,
                         ghost_layers,
                         function_name,
                         assignments=assignments)
    ast.global_variables.update(indexing.index_variables)

    base_pointer_spec = config.base_pointer_specification
    if base_pointer_spec is None:
        base_pointer_spec = []
    base_pointer_info = {f.name: parse_base_pointer_info(base_pointer_spec, [2, 1, 0],
                                                         f.spatial_dimensions, f.index_dimensions)
                         for f in all_fields}

    coord_mapping = {f.name: loop_counter_symbols for f in all_fields}

    if any(FieldType.is_buffer(f) for f in all_fields):
        base_buffer_index = get_base_buffer_index(ast, loop_counter_symbols, iteration_space)
        resolve_buffer_accesses(ast, base_buffer_index, read_only_fields)

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


def created_indexed_cuda_kernel(assignments: NodeCollection, config: CreateKernelConfig):

    index_fields = config.index_fields
    function_name = config.function_name
    coordinate_names = config.coordinate_names
    indexing_creator = indexing_creator_from_params(config.gpu_indexing, config.gpu_indexing_params)
    fields_written = assignments.bound_fields
    fields_read = assignments.rhs_fields

    all_fields = fields_read.union(fields_written)
    read_only_fields = set([f.name for f in fields_read - fields_written])
    # extract the index fields based on the name. The original index field might have been modified
    index_fields = [idx_field for idx_field in index_fields if idx_field.name in [f.name for f in all_fields]]
    non_index_fields = [f for f in all_fields if f not in index_fields]
    spatial_coordinates = {f.spatial_dimensions for f in non_index_fields}
    assert len(spatial_coordinates) == 1, f"Non-index fields do not have the same number of spatial coordinates " \
                                          f"Non index fields are {non_index_fields}, spatial coordinates are " \
                                          f"{spatial_coordinates}"
    spatial_coordinates = list(spatial_coordinates)[0]

    assignments = assignments.all_assignments
    assignments = add_types(assignments, config)

    for index_field in index_fields:
        index_field.field_type = FieldType.INDEXED
        assert FieldType.is_indexed(index_field)
        assert index_field.spatial_dimensions == 1, "Index fields have to be 1D"

    def get_coordinate_symbol_assignment(name):
        for ind_f in index_fields:
            assert isinstance(ind_f.dtype, StructType), "Index fields have to have a struct data type"
            data_type = ind_f.dtype
            if data_type.has_element(name):
                rhs = ind_f[0](name)
                lhs = TypedSymbol(name, data_type.get_element_type(name))
                return SympyAssignment(lhs, rhs)
        raise ValueError(f"Index {name} not found in any of the passed index fields")

    coordinate_symbol_assignments = [get_coordinate_symbol_assignment(n)
                                     for n in coordinate_names[:spatial_coordinates]]
    coordinate_typed_symbols = [eq.lhs for eq in coordinate_symbol_assignments]

    idx_field = list(index_fields)[0]

    iteration_space = normalize_slice(tuple([slice(None, None, None)]) * len(idx_field.spatial_shape),
                                      idx_field.spatial_shape)

    indexing = indexing_creator(iteration_space=iteration_space,
                                data_layout=idx_field.layout)

    function_body = Block(coordinate_symbol_assignments + assignments)
    function_body = indexing.guard(function_body, get_common_field(index_fields).spatial_shape)
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
