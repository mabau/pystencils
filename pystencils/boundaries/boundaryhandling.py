from functools import lru_cache

import numpy as np
import sympy as sp

from pystencils import create_kernel, CreateKernelConfig, Target
from pystencils.astnodes import SympyAssignment
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.boundaries.createindexlist import (
    create_boundary_index_array, numpy_data_type_for_boundary_object)
from pystencils.typing import TypedSymbol, create_type
from pystencils.datahandling.pycuda import PyCudaArrayHandler
from pystencils.field import Field
from pystencils.typing.typed_sympy import FieldPointerSymbol

try:
    # noinspection PyPep8Naming
    import waLBerla as wlb
    if wlb.cpp_available:
        from pystencils.datahandling.parallel_datahandling import ParallelDataHandling
    else:
        ParallelDataHandling = None
except ImportError:
    ParallelDataHandling = None

DEFAULT_FLAG_TYPE = np.uint32


class FlagInterface:
    """Manages the reservation of bits (i.e. flags) in an array of unsigned integers.

    Examples:
        >>> from pystencils import create_data_handling
        >>> dh = create_data_handling((4, 5))
        >>> fi = FlagInterface(dh, 'flag_field', np.uint8)
        >>> assert dh.has_data('flag_field')
        >>> fi.reserve_next_flag()
        2
        >>> fi.reserve_flag(4)
        4
        >>> fi.reserve_next_flag()
        8
    """

    def __init__(self, data_handling, flag_field_name, dtype=DEFAULT_FLAG_TYPE):
        self.flag_field_name = flag_field_name
        self.domain_flag = dtype(1 << 0)
        self._used_flags = {self.domain_flag}
        self.data_handling = data_handling
        self.dtype = dtype
        self.max_bits = self.dtype().itemsize * 8

        # Add flag field to data handling if it does not yet exist
        if data_handling.has_data(self.flag_field_name):
            raise ValueError("There is already a boundary handling registered at the data handling."
                             "If you want to add multiple handling objects, choose a different name.")

        self.flag_field = data_handling.add_array(self.flag_field_name, dtype=self.dtype, cpu=True, gpu=False)
        ff_ghost_layers = data_handling.ghost_layers_of_field(self.flag_field_name)
        for b in data_handling.iterate(ghost_layers=ff_ghost_layers):
            b[self.flag_field_name].fill(self.domain_flag)

    def reserve_next_flag(self):
        for i in range(1, self.max_bits):
            flag = self.dtype(1 << i)
            if flag not in self._used_flags:
                self._used_flags.add(flag)
                assert self._is_power_of_2(flag)
                return flag
        raise ValueError(f"All available {self.max_bits} flags are reserved")

    def reserve_flag(self, flag):
        assert self._is_power_of_2(flag)
        flag = self.dtype(flag)
        if flag in self._used_flags:
            raise ValueError(f"The flag {flag} is already reserved")
        self._used_flags.add(flag)
        return flag

    @staticmethod
    def _is_power_of_2(num):
        return num != 0 and ((num & (num - 1)) == 0)


class BoundaryHandling:

    def __init__(self, data_handling, field_name, stencil, name="boundary_handling", flag_interface=None,
                 target: Target = Target.CPU, openmp=True):
        assert data_handling.has_data(field_name)
        assert data_handling.dim == len(stencil[0]), "Dimension of stencil and data handling do not match"
        self._data_handling = data_handling
        self._field_name = field_name
        self._index_array_name = name + "IndexArrays"
        self._target = target
        self._openmp = openmp
        self._boundary_object_to_boundary_info = {}
        self.stencil = stencil
        self._dirty = True
        fi = flag_interface
        self.flag_interface = fi if fi is not None else FlagInterface(data_handling, name + "Flags")

        if ParallelDataHandling and isinstance(self.data_handling, ParallelDataHandling):
            array_handler = PyCudaArrayHandler()
        else:
            array_handler = self.data_handling.array_handler

        def to_cpu(gpu_version, cpu_version):
            gpu_version = gpu_version.boundary_object_to_index_list
            cpu_version = cpu_version.boundary_object_to_index_list
            for obj, cpu_arr in cpu_version.items():
                array_handler.download(gpu_version[obj], cpu_arr)

        def to_gpu(gpu_version, cpu_version):
            gpu_version = gpu_version.boundary_object_to_index_list
            cpu_version = cpu_version.boundary_object_to_index_list

            for obj, cpu_arr in cpu_version.items():
                if obj not in gpu_version or gpu_version[obj].shape != cpu_arr.shape:
                    gpu_version[obj] = array_handler.to_gpu(cpu_arr)
                else:
                    array_handler.upload(gpu_version[obj], cpu_arr)

        class_ = self.IndexFieldBlockData
        class_.to_cpu = to_cpu
        class_.to_gpu = to_gpu
        gpu = self._target in data_handling._GPU_LIKE_TARGETS
        data_handling.add_custom_class(self._index_array_name, class_, cpu=True, gpu=gpu)

    @property
    def data_handling(self):
        return self._data_handling

    def get_flag(self, boundary_obj):
        return self._boundary_object_to_boundary_info[boundary_obj].flag

    @property
    def shape(self):
        return self._data_handling.shape

    @property
    def dim(self):
        return self._data_handling.dim

    @property
    def boundary_objects(self):
        return tuple(self._boundary_object_to_boundary_info.keys())

    @property
    def flag_array_name(self):
        return self.flag_interface.flag_field_name

    def get_mask(self, slice_obj, boundary_obj, inverse=False):
        if isinstance(boundary_obj, str) and boundary_obj.lower() == 'domain':
            flag = self.flag_interface.domain_flag
        else:
            flag = self._boundary_object_to_boundary_info[boundary_obj].flag

        arr = self.data_handling.gather_array(self.flag_array_name, slice_obj)
        if arr is None:
            return None
        else:
            result = np.bitwise_and(arr, flag)
            if inverse:
                result = np.logical_not(result)
            return result

    def set_boundary(self, boundary_obj, slice_obj=None, mask_callback=None,
                     ghost_layers=True, inner_ghost_layers=True, replace=True, force_flag_value=None):
        """Sets boundary using either a rectangular slice, a boolean mask or a combination of both.

        Args:
            boundary_obj: instance of a boundary object that should be set
            slice_obj: a slice object (can be created with make_slice[]) that selects a part of the domain where
                       the boundary should be set. If none, the complete domain is selected which makes only sense
                       if a mask_callback is passed. The slice can have ':' placeholders, which are interpreted
                       depending on the 'inner_ghost_layers' parameter i.e. if it is True, the slice extends
                       into the ghost layers
            mask_callback: callback function getting x,y (z) parameters of the cell midpoints and returning a
                          boolean mask with True entries where boundary cells should be set.
                          The x, y, z arrays have 2D/3D shape such that they can be used directly
                          to create the boolean return array. i.e return x < 10 sets boundaries in cells with
                          midpoint x coordinate smaller than 10.
            ghost_layers: see DataHandling.iterate()
            inner_ghost_layers: see DataHandling.iterate()
            replace: by default all other flags are erased in the cells where the boundary is set. To add a
                     boundary condition, set this replace flag to False
            force_flag_value: flag that should be reserved for this boundary. Has to be an integer that is a power of 2
                              and was not reserved before for another boundary.
        """
        if isinstance(boundary_obj, str) and boundary_obj.lower() == 'domain':
            flag = self.flag_interface.domain_flag
        else:
            if force_flag_value:
                self.flag_interface.reserve_flag(force_flag_value)
            flag = self._add_boundary(boundary_obj, force_flag_value)

        for b in self._data_handling.iterate(slice_obj, ghost_layers=ghost_layers,
                                             inner_ghost_layers=inner_ghost_layers):
            flag_arr = b[self.flag_interface.flag_field_name]
            if mask_callback is not None:
                mask = mask_callback(*b.midpoint_arrays)
                if replace:
                    flag_arr[mask] = flag
                else:
                    np.bitwise_or(flag_arr, flag, where=mask, out=flag_arr)
                    np.bitwise_and(flag_arr, ~self.flag_interface.domain_flag, where=mask, out=flag_arr)
            else:
                if replace:
                    flag_arr.fill(flag)
                else:
                    np.bitwise_or(flag_arr, flag, out=flag_arr)
                    np.bitwise_and(flag_arr, ~self.flag_interface.domain_flag, out=flag_arr)

        self._dirty = True

        return flag

    def set_boundary_where_flag_is_set(self, boundary_obj, flag):
        """Adds an (additional) boundary to all cells that have been previously marked with the passed flag."""
        self._add_boundary(boundary_obj, flag)
        self._dirty = True
        return flag

    def prepare(self):
        if not self._dirty:
            return
        self._create_index_fields()
        self._dirty = False

    def trigger_reinitialization_of_boundary_data(self, **kwargs):
        if self._dirty:
            self.prepare()
        else:
            ff_ghost_layers = self._data_handling.ghost_layers_of_field(self.flag_interface.flag_field_name)
            for b in self._data_handling.iterate(ghost_layers=ff_ghost_layers):
                for b_obj, setter in b[self._index_array_name].boundary_object_to_data_setter.items():
                    self._boundary_data_initialization(b_obj, setter, **kwargs)

    def __call__(self, **kwargs):
        if self._dirty:
            self.prepare()

        for b in self._data_handling.iterate(gpu=self._target in self._data_handling._GPU_LIKE_TARGETS):
            for b_obj, idx_arr in b[self._index_array_name].boundary_object_to_index_list.items():
                kwargs[self._field_name] = b[self._field_name]
                kwargs['indexField'] = idx_arr
                data_used_in_kernel = (p.fields[0].name
                                       for p in self._boundary_object_to_boundary_info[b_obj].kernel.parameters
                                       if isinstance(p.symbol, FieldPointerSymbol) and p.fields[0].name not in kwargs)
                kwargs.update({name: b[name] for name in data_used_in_kernel})

                self._boundary_object_to_boundary_info[b_obj].kernel(**kwargs)

    def add_fixed_steps(self, fixed_loop, **kwargs):
        if self._dirty:
            self.prepare()

        for b in self._data_handling.iterate(gpu=self._target in self._data_handling._GPU_LIKE_TARGETS):
            for b_obj, idx_arr in b[self._index_array_name].boundary_object_to_index_list.items():
                arguments = kwargs.copy()
                arguments[self._field_name] = b[self._field_name]
                arguments['indexField'] = idx_arr
                data_used_in_kernel = (p.fields[0].name
                                       for p in self._boundary_object_to_boundary_info[b_obj].kernel.parameters
                                       if isinstance(p.symbol, FieldPointerSymbol) and p.field_name not in arguments)
                arguments.update({name: b[name] for name in data_used_in_kernel if name not in arguments})

                kernel = self._boundary_object_to_boundary_info[b_obj].kernel
                fixed_loop.add_call(kernel, arguments)

    def geometry_to_vtk(self, file_name='geometry', boundaries='all', ghost_layers=False):
        """
        Writes a VTK field where each cell with the given boundary is marked with 1, other cells are 0
        This can be used to display the simulation geometry in Paraview

        Params:
            file_name: vtk filename
            boundaries: boundary object, or special string 'domain' for domain cells or special string 'all' for all
                      boundary conditions.
                      can also  be a sequence, to write multiple boundaries to VTK file
            ghost_layers: number of ghost layers to write, or True for all, False for none
        """
        if boundaries == 'all':
            boundaries = list(self._boundary_object_to_boundary_info.keys()) + ['domain']
        elif not hasattr(boundaries, "__len__"):
            boundaries = [boundaries]

        masks_to_name = {}
        for b in boundaries:
            if b == 'domain':
                masks_to_name[self.flag_interface.domain_flag] = 'domain'
            else:
                flag = self._boundary_object_to_boundary_info[b].flag
                masks_to_name[flag] = b.name

        writer = self.data_handling.create_vtk_writer_for_flag_array(file_name, self.flag_interface.flag_field_name,
                                                                     masks_to_name, ghost_layers=ghost_layers)
        writer(1)

    # ------------------------------ Implementation Details ------------------------------------------------------------

    def _add_boundary(self, boundary_obj, flag=None):
        if boundary_obj not in self._boundary_object_to_boundary_info:
            sym_index_field = Field.create_generic('indexField', spatial_dimensions=1,
                                                   dtype=numpy_data_type_for_boundary_object(boundary_obj, self.dim))
            ast = self._create_boundary_kernel(self._data_handling.fields[self._field_name],
                                               sym_index_field, boundary_obj)
            if flag is None:
                flag = self.flag_interface.reserve_next_flag()
            boundary_info = self.BoundaryInfo(boundary_obj, flag=flag, kernel=ast.compile())
            self._boundary_object_to_boundary_info[boundary_obj] = boundary_info
        return self._boundary_object_to_boundary_info[boundary_obj].flag

    def _create_boundary_kernel(self, symbolic_field, symbolic_index_field, boundary_obj):
        return create_boundary_kernel(symbolic_field, symbolic_index_field, self.stencil, boundary_obj,
                                      target=self._target, cpu_openmp=self._openmp)

    def _create_index_fields(self):
        dh = self._data_handling
        ff_ghost_layers = dh.ghost_layers_of_field(self.flag_interface.flag_field_name)
        for b in dh.iterate(ghost_layers=ff_ghost_layers):
            flag_arr = b[self.flag_interface.flag_field_name]
            pdf_arr = b[self._field_name]
            index_array_bd = b[self._index_array_name]
            index_array_bd.clear()
            for b_info in self._boundary_object_to_boundary_info.values():
                boundary_obj = b_info.boundary_object
                idx_arr = create_boundary_index_array(flag_arr, self.stencil, b_info.flag,
                                                      self.flag_interface.domain_flag, boundary_obj,
                                                      ff_ghost_layers, boundary_obj.inner_or_boundary,
                                                      boundary_obj.single_link)
                if idx_arr.size == 0:
                    continue

                boundary_data_setter = BoundaryDataSetter(idx_arr, b.offset, self.stencil, ff_ghost_layers, pdf_arr)
                index_array_bd.boundary_object_to_index_list[b_info.boundary_object] = idx_arr
                index_array_bd.boundary_object_to_data_setter[b_info.boundary_object] = boundary_data_setter
                self._boundary_data_initialization(b_info.boundary_object, boundary_data_setter)

    def _boundary_data_initialization(self, boundary_obj, boundary_data_setter, **kwargs):
        if boundary_obj.additional_data_init_callback:
            boundary_obj.additional_data_init_callback(boundary_data_setter, **kwargs)
        if self._target in self._data_handling._GPU_LIKE_TARGETS:
            self._data_handling.to_gpu(self._index_array_name)

    class BoundaryInfo(object):
        def __init__(self, boundary_obj, flag, kernel):
            self.boundary_object = boundary_obj
            self.flag = flag
            self.kernel = kernel

    class IndexFieldBlockData:
        def __init__(self, *args, **kwargs):
            self.boundary_object_to_index_list = {}
            self.boundary_object_to_data_setter = {}

        def clear(self):
            self.boundary_object_to_index_list.clear()
            self.boundary_object_to_data_setter.clear()


class BoundaryDataSetter:

    def __init__(self, index_array, offset, stencil, ghost_layers, pdf_array):
        self.index_array = index_array
        self.offset = offset
        self.stencil = np.array(stencil)
        self.pdf_array = pdf_array.view()
        self.pdf_array.flags.writeable = False

        arr_field_names = index_array.dtype.names
        self.dim = 3 if 'z' in arr_field_names else 2
        assert 'x' in arr_field_names and 'y' in arr_field_names and 'dir' in arr_field_names, str(arr_field_names)
        self.boundary_data_names = set(self.index_array.dtype.names) - {'x', 'y', 'z', 'dir'}
        self.coord_map = {0: 'x', 1: 'y', 2: 'z'}
        self.ghost_layers = ghost_layers

    def non_boundary_cell_positions(self, coord):
        assert coord < self.dim
        return self.index_array[self.coord_map[coord]] + self.offset[coord] - self.ghost_layers + 0.5

    @lru_cache()
    def link_offsets(self):
        return self.stencil[self.index_array['dir']]

    @lru_cache()
    def link_positions(self, coord):
        return self.non_boundary_cell_positions(coord) + 0.5 * self.link_offsets()[:, coord]

    @lru_cache()
    def boundary_cell_positions(self, coord):
        return self.non_boundary_cell_positions(coord) + self.link_offsets()[:, coord]

    def __setitem__(self, key, value):
        if key not in self.boundary_data_names:
            raise KeyError(f"Invalid boundary data name {key}. Allowed are {self.boundary_data_names}")
        self.index_array[key] = value

    def __getitem__(self, item):
        if item not in self.boundary_data_names:
            raise KeyError(f"Invalid boundary data name {item}. Allowed are {self.boundary_data_names}")
        return self.index_array[item]


class BoundaryOffsetInfo(CustomCodeNode):

    # --------------------------- Functions to be used by boundaries --------------------------

    @staticmethod
    def offset_from_dir(dir_idx, dim):
        return tuple([sp.IndexedBase(symbol, shape=(1,))[dir_idx]
                      for symbol in BoundaryOffsetInfo._offset_symbols(dim)])

    @staticmethod
    def inv_dir(dir_idx):
        return sp.IndexedBase(BoundaryOffsetInfo.INV_DIR_SYMBOL, shape=(1,))[dir_idx]

    # ---------------------------------- Internal ---------------------------------------------

    def __init__(self, stencil):
        dim = len(stencil[0])

        offset_sym = BoundaryOffsetInfo._offset_symbols(dim)
        code = "\n"
        for i in range(dim):
            offset_str = ", ".join([str(d[i]) for d in stencil])
            code += "const int32_t %s [] = { %s };\n" % (offset_sym[i].name, offset_str)

        inv_dirs = []
        for direction in stencil:
            inverse_dir = tuple([-i for i in direction])
            inv_dirs.append(str(stencil.index(inverse_dir)))

        code += "const int32_t %s [] = { %s };\n" % (self.INV_DIR_SYMBOL.name, ", ".join(inv_dirs))
        offset_symbols = BoundaryOffsetInfo._offset_symbols(dim)
        super(BoundaryOffsetInfo, self).__init__(code, symbols_read=set(),
                                                 symbols_defined=set(offset_symbols + [self.INV_DIR_SYMBOL]))

    @staticmethod
    def _offset_symbols(dim):
        return [TypedSymbol(f"c{d}", create_type(np.int32)) for d in ['x', 'y', 'z'][:dim]]

    INV_DIR_SYMBOL = TypedSymbol("invdir", np.int32)


def create_boundary_kernel(field, index_field, stencil, boundary_functor, target=Target.CPU, **kernel_creation_args):
    elements = [BoundaryOffsetInfo(stencil)]
    dir_symbol = TypedSymbol("dir", np.int32)
    elements += [SympyAssignment(dir_symbol, index_field[0]('dir'))]
    elements += boundary_functor(field, direction_symbol=dir_symbol, index_field=index_field)
    config = CreateKernelConfig(index_fields=[index_field], target=target, **kernel_creation_args)
    return create_kernel(elements, config=config)
