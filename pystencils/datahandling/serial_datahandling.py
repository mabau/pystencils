import itertools
import time
from typing import Sequence, Union

import numpy as np

from pystencils.datahandling.blockiteration import SerialBlock
from pystencils.datahandling.datahandling_interface import DataHandling
from pystencils.datahandling.pycuda import PyCudaArrayHandler, PyCudaNotAvailableHandler
from pystencils.enums import Target
from pystencils.field import (
    Field, FieldType, create_numpy_array_with_layout, layout_string_to_tuple,
    spatial_layout_string_to_tuple)
from pystencils.slicing import normalize_slice, remove_ghost_layers
from pystencils.utils import DotDict


class SerialDataHandling(DataHandling):

    def __init__(self,
                 domain_size: Sequence[int],
                 default_ghost_layers: int = 1,
                 default_layout: str = 'SoA',
                 periodicity: Union[bool, Sequence[bool]] = False,
                 default_target: Target = Target.CPU,
                 array_handler=None) -> None:
        """
        Creates a data handling for single node simulations.

        Args:
            domain_size: size of the spatial domain as tuple
            default_ghost_layers: default number of ghost layers used, if not overridden in add_array() method
            default_layout: default layout used, if  not overridden in add_array() method
            default_target: `Target` either 'CPU' or 'GPU'. If set to 'GPU' for each array also a GPU version is
                            allocated if not overwritten in add_array, and synchronization functions are for the GPU by
                            default
        """
        super(SerialDataHandling, self).__init__()
        self._domainSize = tuple(domain_size)
        self.default_ghost_layers = default_ghost_layers
        self.default_layout = default_layout
        self._fields = DotDict()
        self.cpu_arrays = DotDict()
        self.gpu_arrays = DotDict()
        self.custom_data_cpu = DotDict()
        self.custom_data_gpu = DotDict()
        self._custom_data_transfer_functions = {}

        if not array_handler:
            try:
                self.array_handler = PyCudaArrayHandler()
            except Exception:
                self.array_handler = PyCudaNotAvailableHandler()
        else:
            self.array_handler = array_handler

        if periodicity is None or periodicity is False:
            periodicity = [False] * self.dim
        if periodicity is True:
            periodicity = [True] * self.dim

        self._periodicity = periodicity
        self._field_information = {}
        self._default_target = default_target
        self._start_time = time.perf_counter()

    @property
    def default_target(self):
        return self._default_target

    @property
    def dim(self):
        return len(self._domainSize)

    @property
    def shape(self):
        return self._domainSize

    @property
    def periodicity(self):
        return self._periodicity

    @property
    def fields(self):
        return self._fields

    def ghost_layers_of_field(self, name):
        return self._field_information[name]['ghost_layers']

    def values_per_cell(self, name):
        return self._field_information[name]['values_per_cell']

    def add_array(self, name, values_per_cell=1, dtype=np.float64, latex_name=None, ghost_layers=None, layout=None,
                  cpu=True, gpu=None, alignment=False, field_type=FieldType.GENERIC):
        if ghost_layers is None:
            ghost_layers = self.default_ghost_layers
        if layout is None:
            layout = self.default_layout
        if gpu is None:
            gpu = self.default_target in self._GPU_LIKE_TARGETS

        kwargs = {
            'shape': tuple(s + 2 * ghost_layers for s in self._domainSize),
            'dtype': dtype,
        }

        if not hasattr(values_per_cell, '__len__'):
            values_per_cell = (values_per_cell,)
        if len(values_per_cell) == 1 and values_per_cell[0] == 1:
            values_per_cell = ()

        self._field_information[name] = {
            'ghost_layers': ghost_layers,
            'values_per_cell': values_per_cell,
            'layout': layout,
            'dtype': dtype,
            'alignment': alignment,
            'field_type': field_type,
        }

        index_dimensions = len(values_per_cell)
        kwargs['shape'] = kwargs['shape'] + values_per_cell

        if index_dimensions > 0:
            layout_tuple = layout_string_to_tuple(layout, self.dim + index_dimensions)
        else:
            layout_tuple = spatial_layout_string_to_tuple(layout, self.dim)

        # cpu_arr is always created - since there is no create_pycuda_array_with_layout()
        byte_offset = ghost_layers * np.dtype(dtype).itemsize
        cpu_arr = create_numpy_array_with_layout(layout=layout_tuple, alignment=alignment,
                                                 byte_offset=byte_offset, **kwargs)

        if alignment and gpu:
            raise NotImplementedError("Alignment for GPU fields not supported")

        if cpu:
            if name in self.cpu_arrays:
                raise ValueError("CPU Field with this name already exists")
            self.cpu_arrays[name] = cpu_arr
        if gpu:
            if name in self.gpu_arrays:
                raise ValueError("GPU Field with this name already exists")
            self.gpu_arrays[name] = self.array_handler.to_gpu(cpu_arr)

        assert all(f.name != name for f in self.fields.values()), "Symbolic field with this name already exists"
        self.fields[name] = Field.create_from_numpy_array(name, cpu_arr, index_dimensions=index_dimensions,
                                                          field_type=field_type)
        self.fields[name].latex_name = latex_name
        return self.fields[name]

    def add_custom_data(self, name, cpu_creation_function,
                        gpu_creation_function=None, cpu_to_gpu_transfer_func=None, gpu_to_cpu_transfer_func=None):

        if cpu_creation_function and gpu_creation_function:
            if cpu_to_gpu_transfer_func is None or gpu_to_cpu_transfer_func is None:
                raise ValueError("For GPU data, both transfer functions have to be specified")
            self._custom_data_transfer_functions[name] = (cpu_to_gpu_transfer_func, gpu_to_cpu_transfer_func)

        assert name not in self.custom_data_cpu
        if cpu_creation_function:
            assert name not in self.cpu_arrays
            self.custom_data_cpu[name] = cpu_creation_function()

        if gpu_creation_function:
            assert name not in self.gpu_arrays
            self.custom_data_gpu[name] = gpu_creation_function()

    def has_data(self, name):
        return name in self.fields

    def add_array_like(self, name, name_of_template_field, latex_name=None, cpu=True, gpu=None):
        return self.add_array(name, latex_name=latex_name, cpu=cpu, gpu=gpu,
                              **self._field_information[name_of_template_field])

    def iterate(self, slice_obj=None, gpu=False, ghost_layers=True, inner_ghost_layers=True):
        if ghost_layers is True:
            ghost_layers = self.default_ghost_layers
        elif ghost_layers is False:
            ghost_layers = 0
        elif isinstance(ghost_layers, str):
            ghost_layers = self.ghost_layers_of_field(ghost_layers)

        if slice_obj is None:
            slice_obj = (slice(None, None, None),) * self.dim
        slice_obj = normalize_slice(slice_obj, tuple(s + 2 * ghost_layers for s in self._domainSize))
        slice_obj = tuple(s if type(s) is slice else slice(s, s + 1, None) for s in slice_obj)

        arrays = self.gpu_arrays if gpu else self.cpu_arrays
        custom_data_dict = self.custom_data_gpu if gpu else self.custom_data_cpu
        iter_dict = custom_data_dict.copy()
        for name, arr in arrays.items():
            field_gls = self._field_information[name]['ghost_layers']
            if field_gls < ghost_layers:
                continue
            arr = remove_ghost_layers(arr, index_dimensions=len(arr.shape) - self.dim,
                                      ghost_layers=field_gls - ghost_layers)
            iter_dict[name] = arr

        offset = tuple(s.start - ghost_layers for s in slice_obj)
        yield SerialBlock(iter_dict, offset, slice_obj)

    def gather_array(self, name, slice_obj=None, ghost_layers=False, **kwargs):
        gl_to_remove = self._field_information[name]['ghost_layers']
        if isinstance(ghost_layers, int):
            gl_to_remove -= ghost_layers
        if ghost_layers is True:
            gl_to_remove = 0
        arr = self.cpu_arrays[name]
        ind_dimensions = self.fields[name].index_dimensions
        spatial_dimensions = self.fields[name].spatial_dimensions

        arr = remove_ghost_layers(arr, index_dimensions=ind_dimensions, ghost_layers=gl_to_remove)

        if slice_obj is not None:
            normalized_slice = normalize_slice(slice_obj[:spatial_dimensions], arr.shape[:spatial_dimensions])
            normalized_slice = tuple(s if type(s) is slice else slice(s, s + 1, None) for s in normalized_slice)
            normalized_slice += slice_obj[spatial_dimensions:]
            arr = arr[normalized_slice]
        else:
            arr = arr.view()
        arr.flags.writeable = False
        return arr

    def swap(self, name1, name2, gpu=None):
        if gpu is None:
            gpu = self.default_target in self._GPU_LIKE_TARGETS
        arr = self.gpu_arrays if gpu else self.cpu_arrays
        arr[name1], arr[name2] = arr[name2], arr[name1]

    def all_to_cpu(self):
        for name in (self.cpu_arrays.keys() & self.gpu_arrays.keys()) | self._custom_data_transfer_functions.keys():
            self.to_cpu(name)

    def all_to_gpu(self):
        for name in (self.cpu_arrays.keys() & self.gpu_arrays.keys()) | self._custom_data_transfer_functions.keys():
            self.to_gpu(name)

    def run_kernel(self, kernel_function, **kwargs):
        arrays = self.gpu_arrays if kernel_function.ast.backend in self._GPU_LIKE_BACKENDS else self.cpu_arrays
        kernel_function(**{**arrays, **kwargs})

    def get_kernel_kwargs(self, kernel_function, **kwargs):
        result = {}
        result.update(self.gpu_arrays if kernel_function.ast.backend in self._GPU_LIKE_BACKENDS else self.cpu_arrays)
        result.update(kwargs)
        return [result]

    def to_cpu(self, name):
        if name in self._custom_data_transfer_functions:
            transfer_func = self._custom_data_transfer_functions[name][1]
            transfer_func(self.custom_data_gpu[name], self.custom_data_cpu[name])
        else:
            self.array_handler.download(self.gpu_arrays[name], self.cpu_arrays[name])

    def to_gpu(self, name):
        if name in self._custom_data_transfer_functions:
            transfer_func = self._custom_data_transfer_functions[name][0]
            transfer_func(self.custom_data_gpu[name], self.custom_data_cpu[name])
        else:
            self.array_handler.upload(self.gpu_arrays[name], self.cpu_arrays[name])

    def is_on_gpu(self, name):
        return name in self.gpu_arrays

    def synchronization_function_cpu(self, names, stencil_name=None, **_):
        return self.synchronization_function(names, stencil_name, target=Target.CPU)

    def synchronization_function_gpu(self, names, stencil_name=None, **_):
        return self.synchronization_function(names, stencil_name, target=Target.GPU)

    def synchronization_function(self, names, stencil=None, target=None, functor=None, **_):
        if target is None:
            target = self.default_target
        assert target in (Target.CPU, Target.GPU)
        if not hasattr(names, '__len__') or type(names) is str:
            names = [names]

        filtered_stencil = []
        neighbors = [-1, 0, 1]

        if (stencil is None and self.dim == 2) or (stencil is not None and stencil.startswith('D2')):
            directions = itertools.product(*[neighbors] * 2)
        elif (stencil is None and self.dim == 3) or (stencil is not None and stencil.startswith('D3')):
            directions = itertools.product(*[neighbors] * 3)
        else:
            raise ValueError("Invalid stencil")

        for direction in directions:
            use_direction = True
            if direction == (0, 0) or direction == (0, 0, 0):
                use_direction = False
            for component, periodicity in zip(direction, self._periodicity):
                if not periodicity and component != 0:
                    use_direction = False
            if use_direction:
                filtered_stencil.append(direction)

        result = []
        for name in names:
            gls = self._field_information[name]['ghost_layers']
            values_per_cell = self._field_information[name]['values_per_cell']
            if values_per_cell == ():
                values_per_cell = (1,)
            if len(values_per_cell) == 1:
                values_per_cell = values_per_cell[0]

            if len(filtered_stencil) > 0:
                if target == Target.CPU:
                    if functor is None:
                        from pystencils.slicing import get_periodic_boundary_functor
                        functor = get_periodic_boundary_functor
                    result.append(functor(filtered_stencil, ghost_layers=gls))
                else:
                    if functor is None:
                        from pystencils.gpucuda.periodicity import get_periodic_boundary_functor as functor
                        target = Target.GPU
                    result.append(functor(filtered_stencil, self._domainSize,
                                          index_dimensions=self.fields[name].index_dimensions,
                                          index_dim_shape=values_per_cell,
                                          dtype=self.fields[name].dtype.numpy_dtype,
                                          ghost_layers=gls,
                                          target=target))

        if target == Target.CPU:
            def result_functor():
                for arr_name, func in zip(names, result):
                    func(pdfs=self.cpu_arrays[arr_name])
        else:
            def result_functor():
                for arr_name, func in zip(names, result):
                    func(pdfs=self.gpu_arrays[arr_name])

        return result_functor

    @property
    def array_names(self):
        return tuple(self.fields.keys())

    @property
    def custom_data_names(self):
        return tuple(self.custom_data_cpu.keys())

    def reduce_float_sequence(self, sequence, operation, all_reduce=False) -> np.array:
        return np.array(sequence)

    def reduce_int_sequence(self, sequence, operation, all_reduce=False) -> np.array:
        return np.array(sequence)

    def create_vtk_writer(self, file_name, data_names, ghost_layers=False):
        from pystencils.datahandling.vtk import image_to_vtk

        def writer(step):
            full_file_name = "%s_%08d" % (file_name, step,)
            cell_data = {}
            for name in data_names:
                field = self._get_field_with_given_number_of_ghost_layers(name, ghost_layers)
                if self.dim == 2:
                    field = field[:, :, np.newaxis]
                if len(field.shape) == 3:
                    cell_data[name] = np.ascontiguousarray(field)
                elif len(field.shape) == 4:
                    values_per_cell = field.shape[-1]
                    if values_per_cell == self.dim:
                        field = [np.ascontiguousarray(field[..., i]) for i in range(values_per_cell)]
                        if len(field) == 2:
                            field.append(np.zeros_like(field[0]))
                        cell_data[name] = tuple(field)
                    else:
                        for i in range(values_per_cell):
                            cell_data["%s[%d]" % (name, i)] = np.ascontiguousarray(field[..., i])
                else:
                    raise NotImplementedError("VTK export for fields with more than one index "
                                              "coordinate not implemented")
            image_to_vtk(full_file_name, cell_data=cell_data)

        return writer

    def create_vtk_writer_for_flag_array(self, file_name, data_name, masks_to_name, ghost_layers=False):
        from pystencils.datahandling.vtk import image_to_vtk

        def writer(step):
            full_file_name = "%s_%08d" % (file_name, step,)
            field = self._get_field_with_given_number_of_ghost_layers(data_name, ghost_layers)
            if self.dim == 2:
                field = field[:, :, np.newaxis]
            cell_data = {name: np.ascontiguousarray(np.bitwise_and(field, field.dtype.type(mask)) > 0, dtype=np.uint8)
                         for mask, name in masks_to_name.items()}
            image_to_vtk(full_file_name, cell_data=cell_data)

        return writer

    def _get_field_with_given_number_of_ghost_layers(self, name, ghost_layers):
        actual_ghost_layers = self.ghost_layers_of_field(name)
        if ghost_layers is True:
            ghost_layers = actual_ghost_layers

        gl_to_remove = actual_ghost_layers - ghost_layers
        ind_dims = len(self._field_information[name]['values_per_cell'])
        return remove_ghost_layers(self.cpu_arrays[name], ind_dims, gl_to_remove)

    def log(self, *args, level='INFO'):
        level = level.upper()
        message = " ".join(str(e) for e in args)

        time_running = time.perf_counter() - self._start_time
        spacing = 7 - len(str(int(time_running)))
        message = f"[{level: <8}]{spacing * '-'}({time_running:.3f} sec) {message} "
        print(message, flush=True)

    def log_on_root(self, *args, level='INFO'):
        self.log(*args, level=level)

    @property
    def is_root(self):
        return True

    @property
    def world_rank(self):
        return 0

    def save_all(self, file):
        np.savez_compressed(file, **self.cpu_arrays)

    def load_all(self, file):
        if '.npz' not in file:
            file += '.npz'
        file_contents = np.load(file)
        for arr_name, arr_contents in self.cpu_arrays.items():
            if arr_name not in file_contents:
                print(f"Skipping read data {arr_name} because there is no data with this name in data handling")
                continue
            if file_contents[arr_name].shape != arr_contents.shape:
                print(f"Skipping read data {arr_name} because shapes don't match. "
                      f"Read array shape {file_contents[arr_name].shape}, existing array shape {arr_contents.shape}")
                continue
            np.copyto(arr_contents, file_contents[arr_name])
