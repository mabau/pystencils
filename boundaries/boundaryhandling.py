import numpy as np
import sympy as sp
from pystencils.assignment import Assignment
from pystencils import Field, TypedSymbol, create_indexed_kernel
from pystencils.backends.cbackend import CustomCppCode
from pystencils.boundaries.createindexlist import numpyDataTypeForBoundaryObject, createBoundaryIndexArray
from pystencils.cache import memorycache
from pystencils.data_types import create_type


class FlagInterface:
    FLAG_DTYPE = np.uint32

    def __init__(self, data_handling, flag_field_name):
        self.flag_field_name = flag_field_name
        self.domain_flag = self.FLAG_DTYPE(1 << 0)
        self._nextFreeFlag = 1
        self.data_handling = data_handling

        # Add flag field to data handling if it does not yet exist
        if data_handling.has_data(self.flag_field_name):
            raise ValueError("There is already a boundary handling registered at the data handling."
                             "If you want to add multiple handlings, choose a different name.")

        data_handling.add_array(self.flag_field_name, dtype=self.FLAG_DTYPE, cpu=True, gpu=False)
        ff_ghost_layers = data_handling.ghost_layers_of_field(self.flag_field_name)
        for b in data_handling.iterate(ghost_layers=ff_ghost_layers):
            b[self.flag_field_name].fill(self.domain_flag)

    def allocate_next_flag(self):
        result = self.FLAG_DTYPE(1 << self._nextFreeFlag)
        self._nextFreeFlag += 1
        return result


class BoundaryHandling:

    def __init__(self, data_handling, field_name, stencil, name="boundary_handling", flag_interface=None,
                 target='cpu', openmp=True):
        assert data_handling.has_data(field_name)

        self._data_handling = data_handling
        self._field_name = field_name
        self._index_array_name = name + "IndexArrays"
        self._target = target
        self._openmp = openmp
        self._boundary_object_to_boundary_info = {}
        self.stencil = stencil
        self._dirty = True
        self.flag_interface = flag_interface if flag_interface is not None else FlagInterface(data_handling, name + "Flags")

        gpu = self._target == 'gpu'
        data_handling.add_custom_class(self._index_array_name, self.IndexFieldBlockData, cpu=True, gpu=gpu)

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
        return tuple(self._boundaryObjectToName.keys())

    @property
    def flag_array_name(self):
        return self.flag_interface.flag_field_name

    def get_boundary_name_to_flag_dict(self):
        result = {bObj.name: bInfo.flag for bObj, bInfo in self._boundary_object_to_boundary_info.items()}
        result['domain'] = self.flag_interface.domain_flag
        return result

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
                     ghost_layers=True, inner_ghost_layers=True, replace=True):
        """
        Sets boundary using either a rectangular slice, a boolean mask or a combination of both

        :param boundary_obj: instance of a boundary object that should be set
        :param slice_obj: a slice object (can be created with makeSlice[]) that selects a part of the domain where
                          the boundary should be set. If none, the complete domain is selected which makes only sense
                          if a maskCallback is passed. The slice can have ':' placeholders, which are interpreted
                          depending on the 'includeGhostLayers' parameter i.e. if it is True, the slice extends
                          into the ghost layers
        :param mask_callback: callback function getting x,y (z) parameters of the cell midpoints and returning a
                             boolean mask with True entries where boundary cells should be set.
                             The x, y, z arrays have 2D/3D shape such that they can be used directly
                             to create the boolean return array. i.e return x < 10 sets boundaries in cells with
                             midpoint x coordinate smaller than 10.
        :param ghost_layers see DataHandling.iterate()
        """
        if isinstance(boundary_obj, str) and boundary_obj.lower() == 'domain':
            flag = self.flag_interface.domain_flag
        else:
            flag = self._add_boundary(boundary_obj)

        for b in self._data_handling.iterate(slice_obj, ghost_layers=ghost_layers, inner_ghost_layers=inner_ghost_layers):
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
                for bObj, setter in b[self._index_array_name].boundaryObjectToDataSetter.items():
                    self._boundary_data_initialization(bObj, setter, **kwargs)

    def __call__(self, **kwargs):
        if self._dirty:
            self.prepare()

        for b in self._data_handling.iterate(gpu=self._target == 'gpu'):
            for bObj, idxArr in b[self._index_array_name].boundary_object_to_index_list.items():
                kwargs[self._field_name] = b[self._field_name]
                kwargs['indexField'] = idxArr
                data_used_in_kernel = (p.field_name
                                       for p in self._boundary_object_to_boundary_info[bObj].kernel.parameters
                                       if p.isFieldPtrArgument and p.field_name not in kwargs)
                kwargs.update({name: b[name] for name in data_used_in_kernel})

                self._boundary_object_to_boundary_info[bObj].kernel(**kwargs)

    def geometry_to_vtk(self, file_name='geometry', boundaries='all', ghost_layers=False):
        """
        Writes a VTK field where each cell with the given boundary is marked with 1, other cells are 0
        This can be used to display the simulation geometry in Paraview
        :param file_name: vtk filename
        :param boundaries: boundary object, or special string 'domain' for domain cells or special string 'all' for all
                         boundary conditions.
                         can also  be a sequence, to write multiple boundaries to VTK file
        :param ghost_layers: number of ghost layers to write, or True for all, False for none
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
                masks_to_name[self._boundary_object_to_boundary_info[b].flag] = b.name

        writer = self.data_handling.create_vtk_writer_for_flag_array(file_name, self.flag_interface.flag_field_name,
                                                                     masks_to_name, ghost_layers=ghost_layers)
        writer(1)

    # ------------------------------ Implementation Details ------------------------------------------------------------

    def _add_boundary(self, boundary_obj, flag=None):
        if boundary_obj not in self._boundary_object_to_boundary_info:
            symbolic_index_field = Field.create_generic('indexField', spatial_dimensions=1,
                                                        dtype=numpyDataTypeForBoundaryObject(boundary_obj, self.dim))
            ast = self._create_boundary_kernel(self._data_handling.fields[self._field_name],
                                               symbolic_index_field, boundary_obj)
            if flag is None:
                flag = self.flag_interface.allocate_next_flag()
            boundary_info = self.BoundaryInfo(boundary_obj, flag=flag, kernel=ast.compile())
            self._boundary_object_to_boundary_info[boundary_obj] = boundary_info
        return self._boundary_object_to_boundary_info[boundary_obj].flag

    def _create_boundary_kernel(self, symbolic_field, symbolic_index_field, boundary_obj):
        return create_boundary_kernel(symbolic_field, symbolic_index_field, self.stencil, boundary_obj,
                                      target=self._target, openmp=self._openmp)

    def _create_index_fields(self):
        dh = self._data_handling
        ff_ghost_layers = dh.ghost_layers_of_field(self.flag_interface.flag_field_name)
        for b in dh.iterate(ghost_layers=ff_ghost_layers):
            flag_arr = b[self.flag_interface.flag_field_name]
            pdf_arr = b[self._field_name]
            index_array_bd = b[self._index_array_name]
            index_array_bd.clear()
            for bInfo in self._boundary_object_to_boundary_info.values():
                idxArr = createBoundaryIndexArray(flag_arr, self.stencil, bInfo.flag, self.flag_interface.domain_flag,
                                                  bInfo.boundaryObject, ff_ghost_layers)
                if idxArr.size == 0:
                    continue

                boundary_data_setter = BoundaryDataSetter(idxArr, b.offset, self.stencil, ff_ghost_layers, pdf_arr)
                index_array_bd.boundary_object_to_index_list[bInfo.boundaryObject] = idxArr
                index_array_bd.boundaryObjectToDataSetter[bInfo.boundaryObject] = boundary_data_setter
                self._boundary_data_initialization(bInfo.boundaryObject, boundary_data_setter)

    def _boundary_data_initialization(self, boundary_obj, boundary_data_setter, **kwargs):
        if boundary_obj.additional_data_init_callback:
            boundary_obj.additional_data_init_callback(boundary_data_setter, **kwargs)
        if self._target == 'gpu':
            self._data_handling.to_gpu(self._index_array_name)

    class BoundaryInfo(object):
        def __init__(self, boundary_obj, flag, kernel):
            self.boundaryObject = boundary_obj
            self.flag = flag
            self.kernel = kernel

    class IndexFieldBlockData:
        def __init__(self, *args, **kwargs):
            self.boundary_object_to_index_list = {}
            self.boundaryObjectToDataSetter = {}

        def clear(self):
            self.boundary_object_to_index_list.clear()
            self.boundaryObjectToDataSetter.clear()

        @staticmethod
        def to_cpu(gpu_version, cpu_version):
            gpu_version = gpu_version.boundary_object_to_index_list
            cpu_version = cpu_version.boundary_object_to_index_list
            for obj, cpuArr in cpu_version.values():
                gpu_version[obj].get(cpuArr)

        @staticmethod
        def to_gpu(gpu_version, cpu_version):
            from pycuda import gpuarray
            gpu_version = gpu_version.boundary_object_to_index_list
            cpu_version = cpu_version.boundary_object_to_index_list
            for obj, cpuArr in cpu_version.items():
                if obj not in gpu_version:
                    gpu_version[obj] = gpuarray.to_gpu(cpuArr)
                else:
                    gpu_version[obj].set(cpuArr)


class BoundaryDataSetter:

    def __init__(self, index_array, offset, stencil, ghost_layers, pdf_array):
        self.indexArray = index_array
        self.offset = offset
        self.stencil = np.array(stencil)
        self.pdf_array = pdf_array.view()
        self.pdf_array.flags.writeable = False

        arr_field_names = index_array.dtype.names
        self.dim = 3 if 'z' in arr_field_names else 2
        assert 'x' in arr_field_names and 'y' in arr_field_names and 'dir' in arr_field_names, str(arr_field_names)
        self.boundary_data_names = set(self.indexArray.dtype.names) - set(['x', 'y', 'z', 'dir'])
        self.coord_map = {0: 'x', 1: 'y', 2: 'z'}
        self.ghost_layers = ghost_layers

    def non_boundary_cell_positions(self, coord):
        assert coord < self.dim
        return self.indexArray[self.coord_map[coord]] + self.offset[coord] - self.ghost_layers + 0.5

    @memorycache()
    def link_offsets(self):
        return self.stencil[self.indexArray['dir']]

    @memorycache()
    def link_positions(self, coord):
        return self.non_boundary_cell_positions(coord) + 0.5 * self.link_offsets()[:, coord]

    @memorycache()
    def boundary_cell_positions(self, coord):
        return self.non_boundary_cell_positions(coord) + self.link_offsets()[:, coord]

    def __setitem__(self, key, value):
        if key not in self.boundary_data_names:
            raise KeyError("Invalid boundary data name %s. Allowed are %s" % (key, self.boundary_data_names))
        self.indexArray[key] = value

    def __getitem__(self, item):
        if item not in self.boundary_data_names:
            raise KeyError("Invalid boundary data name %s. Allowed are %s" % (item, self.boundary_data_names))
        return self.indexArray[item]


class BoundaryOffsetInfo(CustomCppCode):

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
            code += "const int64_t %s [] = { %s };\n" % (offset_sym[i].name, offset_str)

        inv_dirs = []
        for direction in stencil:
            inverse_dir = tuple([-i for i in direction])
            inv_dirs.append(str(stencil.index(inverse_dir)))

        code += "const int %s [] = { %s };\n" % (self.INV_DIR_SYMBOL.name, ", ".join(inv_dirs))
        offset_symbols = BoundaryOffsetInfo._offset_symbols(dim)
        super(BoundaryOffsetInfo, self).__init__(code, symbols_read=set(),
                                                 symbols_defined=set(offset_symbols + [self.INV_DIR_SYMBOL]))

    @staticmethod
    def _offset_symbols(dim):
        return [TypedSymbol("c_%d" % (d,), create_type(np.int64)) for d in range(dim)]

    INV_DIR_SYMBOL = TypedSymbol("inv_dir", "int")


def create_boundary_kernel(field, index_field, stencil, boundary_functor, target='cpu', openmp=True):
    elements = [BoundaryOffsetInfo(stencil)]
    index_arr_dtype = index_field.dtype.numpy_dtype
    dir_symbol = TypedSymbol("dir", index_arr_dtype.fields['dir'][0])
    elements += [Assignment(dir_symbol, index_field[0]('dir'))]
    elements += boundary_functor(field, directionSymbol=dir_symbol, indexField=index_field)
    return create_indexed_kernel(elements, [index_field], target=target, cpu_openmp=openmp)
