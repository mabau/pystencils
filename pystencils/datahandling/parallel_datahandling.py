import os
import warnings

import numpy as np
# noinspection PyPep8Naming
import waLBerla as wlb

from pystencils.datahandling.blockiteration import block_iteration, sliced_block_iteration
from pystencils.datahandling.datahandling_interface import DataHandling
from pystencils.enums import Backend
from pystencils.field import Field, FieldType
from pystencils.typing.typed_sympy import FieldPointerSymbol
from pystencils.utils import DotDict
from pystencils import Target


class ParallelDataHandling(DataHandling):
    GPU_DATA_PREFIX = "gpu_"
    VTK_COUNTER = 0

    def __init__(self, blocks, default_ghost_layers=1, default_layout='SoA', dim=3, default_target=Target.CPU):
        """
        Creates data handling based on walberla block storage

        Args:
            blocks: walberla block storage
            default_ghost_layers: nr of ghost layers used if not specified in add() method
            default_layout: layout used if no layout is given to add() method
            dim: dimension of scenario,
                 walberla always uses three dimensions, so if dim=2 the extend of the
                 z coordinate of blocks has to be 1
            default_target: `Target`, either 'CPU' or 'GPU' . If set to 'GPU' for each array also a GPU version is
                            allocated if not overwritten in add_array, and synchronization functions are for the GPU by
                            default
        """
        super(ParallelDataHandling, self).__init__()
        assert dim in (2, 3)
        self._blocks = blocks
        self._default_ghost_layers = default_ghost_layers
        self._default_layout = default_layout
        self._fields = DotDict()  # maps name to symbolic pystencils field
        self._field_name_to_cpu_data_name = {}
        self._field_name_to_gpu_data_name = {}
        self._data_names = set()
        self._dim = dim
        self._fieldInformation = {}
        self._cpu_gpu_pairs = []
        self._custom_data_transfer_functions = {}
        self._custom_data_names = []
        self._reduce_map = {
            'sum': wlb.mpi.SUM,
            'min': wlb.mpi.MIN,
            'max': wlb.mpi.MAX,
        }

        if self._dim == 2:
            assert self.blocks.getDomainCellBB().size[2] == 1
        self._default_target = default_target

    @property
    def default_target(self):
        return self._default_target

    @property
    def dim(self):
        return self._dim

    @property
    def shape(self):
        return self.blocks.getDomainCellBB().size[:self.dim]

    @property
    def periodicity(self):
        return self.blocks.periodic[:self._dim]

    @property
    def fields(self):
        return self._fields

    @property
    def blocks(self):
        return self._blocks

    @property
    def default_ghost_layers(self):
        return self._default_ghost_layers

    @property
    def default_layout(self):
        return self._default_layout

    @property
    def data_names(self):
        return self.data_names

    def ghost_layers_of_field(self, name):
        return self._fieldInformation[name]['ghost_layers']

    def values_per_cell(self, name):
        return self._fieldInformation[name]['values_per_cell']

    def add_custom_data(self, name, cpu_creation_function,
                        gpu_creation_function=None, cpu_to_gpu_transfer_func=None, gpu_to_cpu_transfer_func=None):
        if cpu_creation_function and gpu_creation_function:
            if cpu_to_gpu_transfer_func is None or gpu_to_cpu_transfer_func is None:
                raise ValueError("For GPU data, both transfer functions have to be specified")
            self._custom_data_transfer_functions[name] = (cpu_to_gpu_transfer_func, gpu_to_cpu_transfer_func)

        if cpu_creation_function:
            self.blocks.addBlockData(name, cpu_creation_function)
        if gpu_creation_function:
            self.blocks.addBlockData(self.GPU_DATA_PREFIX + name, gpu_creation_function)
        self._custom_data_names.append(name)

    def add_array(self, name, values_per_cell=1, dtype=np.float64, latex_name=None, ghost_layers=None,
                  layout=None, cpu=True, gpu=None, alignment=False, field_type=FieldType.GENERIC):
        if ghost_layers is None:
            ghost_layers = self.default_ghost_layers
        if gpu is None:
            gpu = self.default_target == Target.GPU
        if layout is None:
            layout = self.default_layout
        if len(self.blocks) == 0:
            raise ValueError("Data handling expects that each process has at least one block")
        if hasattr(dtype, 'type'):
            dtype = dtype.type
        if name in self.blocks[0].fieldNames or self.GPU_DATA_PREFIX + name in self.blocks[0].fieldNames:
            raise ValueError("Data with this name has already been added")

        if alignment is False or alignment is None:
            alignment = 0
        if hasattr(values_per_cell, '__len__'):
            raise NotImplementedError("Parallel data handling does not support multiple index dimensions")

        self._fieldInformation[name] = {
            'ghost_layers': ghost_layers,
            'values_per_cell': values_per_cell,
            'layout': layout,
            'dtype': dtype,
            'alignment': alignment,
            'field_type': field_type,
        }

        layout_map = {'fzyx': wlb.field.Layout.fzyx, 'zyxf': wlb.field.Layout.zyxf,
                      'f': wlb.field.Layout.fzyx,
                      'SoA': wlb.field.Layout.fzyx, 'AoS': wlb.field.Layout.zyxf}

        if cpu:
            wlb.field.addToStorage(self.blocks, name, dtype, fSize=values_per_cell, layout=layout_map[layout],
                                   ghostLayers=ghost_layers, alignment=alignment)
        if gpu:
            if alignment != 0:
                raise ValueError("Alignment for walberla GPU fields not yet supported")
            wlb.cuda.addGpuFieldToStorage(self.blocks, self.GPU_DATA_PREFIX + name, dtype, fSize=values_per_cell,
                                          usePitchedMem=False, ghostLayers=ghost_layers, layout=layout_map[layout])

        if cpu and gpu:
            self._cpu_gpu_pairs.append((name, self.GPU_DATA_PREFIX + name))

        block_bb = self.blocks.getBlockCellBB(self.blocks[0])
        shape = tuple(s + 2 * ghost_layers for s in block_bb.size[:self.dim])
        index_dimensions = 1 if values_per_cell > 1 else 0
        if index_dimensions == 1:
            shape += (values_per_cell,)

        assert all(f.name != name for f in self.fields.values()), "Symbolic field with this name already exists"

        self.fields[name] = Field.create_generic(name, self.dim, dtype, index_dimensions, layout,
                                                 index_shape=(values_per_cell,) if index_dimensions > 0 else None,
                                                 field_type=field_type)
        self.fields[name].latex_name = latex_name
        self._field_name_to_cpu_data_name[name] = name
        if gpu:
            self._field_name_to_gpu_data_name[name] = self.GPU_DATA_PREFIX + name
        return self.fields[name]

    def has_data(self, name):
        return name in self._fields

    @property
    def array_names(self):
        return tuple(self.fields.keys())

    @property
    def custom_data_names(self):
        return tuple(self._custom_data_names)

    def add_array_like(self, name, name_of_template_field, latex_name=None, cpu=True, gpu=None):
        return self.add_array(name, latex_name=latex_name, cpu=cpu, gpu=gpu,
                              **self._fieldInformation[name_of_template_field])

    def swap(self, name1, name2, gpu=False):
        if gpu:
            name1 = self.GPU_DATA_PREFIX + name1
            name2 = self.GPU_DATA_PREFIX + name2
        for block in self.blocks:
            block[name1].swapDataPointers(block[name2])

    def iterate(self, slice_obj=None, gpu=False, ghost_layers=True, inner_ghost_layers=True):
        if ghost_layers is True:
            ghost_layers = self.default_ghost_layers
        elif ghost_layers is False:
            ghost_layers = 0
        elif isinstance(ghost_layers, str):
            ghost_layers = self.ghost_layers_of_field(ghost_layers)

        if inner_ghost_layers is True:
            inner_ghost_layers = self.default_ghost_layers
        elif inner_ghost_layers is False:
            inner_ghost_layers = 0
        elif isinstance(ghost_layers, str):
            ghost_layers = self.ghost_layers_of_field(ghost_layers)

        prefix = self.GPU_DATA_PREFIX if gpu else ""
        if slice_obj is not None:
            yield from sliced_block_iteration(self.blocks, slice_obj, inner_ghost_layers, ghost_layers,
                                              self.dim, prefix)
        else:
            yield from block_iteration(self.blocks, ghost_layers, self.dim, prefix)

    def gather_array(self, name, slice_obj=None, all_gather=False, ghost_layers=False):
        if ghost_layers is not False:
            warnings.warn("gather_array with ghost layers is only supported in serial data handling. "
                          "Array without ghost layers is returned")

        if slice_obj is None:
            slice_obj = tuple([slice(None, None, None)] * self.dim)
        if self.dim == 2:
            slice_obj = slice_obj[:2] + (0.5,) + slice_obj[2:]

        last_element = slice_obj[3:]

        array = wlb.field.gatherField(self.blocks, name, slice_obj[:3], all_gather)
        if array is None:
            return None

        if self.dim == 2:
            array = array[:, :, 0]
        if last_element and self.fields[name].index_dimensions > 0:
            array = array[..., last_element[0]]

        return array

    def _normalize_arr_shape(self, arr, index_dimensions):
        if index_dimensions == 0 and len(arr.shape) > 3:
            arr = arr[..., 0]
        if self.dim == 2 and len(arr.shape) > 2:
            arr = arr[:, :, 0]
        return arr

    def run_kernel(self, kernel_function, **kwargs):
        for arg_dict in self.get_kernel_kwargs(kernel_function, **kwargs):
            kernel_function(**arg_dict)

    def get_kernel_kwargs(self, kernel_function, **kwargs):
        if kernel_function.ast.backend == Backend.CUDA:
            name_map = self._field_name_to_gpu_data_name
            to_array = wlb.cuda.toGpuArray
        else:
            name_map = self._field_name_to_cpu_data_name
            to_array = wlb.field.toArray
        data_used_in_kernel = [(name_map[p.symbol.field_name], self.fields[p.symbol.field_name])
                               for p in kernel_function.parameters if
                               isinstance(p.symbol, FieldPointerSymbol) and p.symbol.field_name not in kwargs]

        result = []
        for block in self.blocks:
            field_args = {}
            for data_name, f in data_used_in_kernel:
                arr = to_array(block[data_name], with_ghost_layers=[True, True, self.dim == 3])
                arr = self._normalize_arr_shape(arr, f.index_dimensions)
                field_args[f.name] = arr
            field_args.update(kwargs)
            result.append(field_args)
        return result

    def to_cpu(self, name):
        if name in self._custom_data_transfer_functions:
            transfer_func = self._custom_data_transfer_functions[name][1]
            for block in self.blocks:
                transfer_func(block[self.GPU_DATA_PREFIX + name], block[name])
        else:
            wlb.cuda.copyFieldToCpu(self.blocks, self.GPU_DATA_PREFIX + name, name)

    def to_gpu(self, name):
        if name in self._custom_data_transfer_functions:
            transfer_func = self._custom_data_transfer_functions[name][0]
            for block in self.blocks:
                transfer_func(block[self.GPU_DATA_PREFIX + name], block[name])
        else:
            wlb.cuda.copyFieldToGpu(self.blocks, self.GPU_DATA_PREFIX + name, name)

    def is_on_gpu(self, name):
        return (name, self.GPU_DATA_PREFIX + name) in self._cpu_gpu_pairs

    def all_to_cpu(self):
        for cpu_name, gpu_name in self._cpu_gpu_pairs:
            wlb.cuda.copyFieldToCpu(self.blocks, gpu_name, cpu_name)
        for name in self._custom_data_transfer_functions.keys():
            self.to_cpu(name)

    def all_to_gpu(self):
        for cpu_name, gpu_name in self._cpu_gpu_pairs:
            wlb.cuda.copyFieldToGpu(self.blocks, gpu_name, cpu_name)
        for name in self._custom_data_transfer_functions.keys():
            self.to_gpu(name)

    def synchronization_function_cpu(self, names, stencil=None, buffered=True, stencil_restricted=False, **_):
        return self.synchronization_function(names, stencil, Target.CPU, buffered, stencil_restricted)

    def synchronization_function_gpu(self, names, stencil=None, buffered=True, stencil_restricted=False, **_):
        return self.synchronization_function(names, stencil, Target.GPU, buffered, stencil_restricted)

    def synchronization_function(self, names, stencil=None, target=None, buffered=True, stencil_restricted=False):
        if target is None:
            target = self.default_target

        if stencil is None:
            stencil = 'D3Q27' if self.dim == 3 else 'D2Q9'

        if not hasattr(names, '__len__') or type(names) is str:
            names = [names]

        create_scheme = wlb.createUniformBufferedScheme if buffered else wlb.createUniformDirectScheme
        if target == Target.CPU:
            create_packing = wlb.field.createPackInfo if buffered else wlb.field.createMPIDatatypeInfo
            if buffered and stencil_restricted:
                create_packing = wlb.field.createStencilRestrictedPackInfo
        else:
            assert target == Target.GPU
            create_packing = wlb.cuda.createPackInfo if buffered else wlb.cuda.createMPIDatatypeInfo
            names = [self.GPU_DATA_PREFIX + name for name in names]

        sync_function = create_scheme(self.blocks, stencil)
        for name in names:
            sync_function.addDataToCommunicate(create_packing(self.blocks, name))

        return sync_function

    def reduce_float_sequence(self, sequence, operation, all_reduce=False):
        if all_reduce:
            return np.array(wlb.mpi.allreduceReal(sequence, self._reduce_map[operation.lower()]))
        else:
            result = np.array(wlb.mpi.reduceReal(sequence, self._reduce_map[operation.lower()], 0))
            return result if wlb.mpi.worldRank() == 0 else None

    def reduce_int_sequence(self, sequence, operation, all_reduce=False):
        if all_reduce:
            return np.array(wlb.mpi.allreduceInt(sequence, self._reduce_map[operation.lower()]))
        else:
            result = np.array(wlb.mpi.reduceInt(sequence, self._reduce_map[operation.lower()], 0))
            return result if wlb.mpi.worldRank() == 0 else None

    def create_vtk_writer(self, file_name, data_names, ghost_layers=False):
        if ghost_layers is False:
            ghost_layers = 0
        if ghost_layers is True:
            ghost_layers = min(self.ghost_layers_of_field(n) for n in data_names)
        file_name = "%s_%02d" % (file_name, ParallelDataHandling.VTK_COUNTER)
        ParallelDataHandling.VTK_COUNTER += 1
        output = wlb.vtk.makeOutput(self.blocks, file_name, ghostLayers=ghost_layers)
        for n in data_names:
            output.addCellDataWriter(wlb.field.createVTKWriter(self.blocks, n))
        return output

    def create_vtk_writer_for_flag_array(self, file_name, data_name, masks_to_name, ghost_layers=False):
        if ghost_layers is False:
            ghost_layers = 0
        if ghost_layers is True:
            ghost_layers = self.ghost_layers_of_field(data_name)

        output = wlb.vtk.makeOutput(self.blocks, file_name, ghostLayers=ghost_layers)
        for mask, name in masks_to_name.items():
            w = wlb.field.createBinarizationVTKWriter(self.blocks, data_name, mask, name)
            output.addCellDataWriter(w)
        return output

    @staticmethod
    def log(*args, level='INFO'):
        _log_map = {
            'DEVEL': wlb.log_devel,
            'RESULT': wlb.log_result,
            'INFO': wlb.log_info,
            'WARNING': wlb.log_warning,
            'PROGRESS': wlb.log_progress,
        }
        level = level.upper()
        message = " ".join(str(e) for e in args)
        _log_map[level](message)

    def log_on_root(self, *args, level='INFO'):
        if self.is_root:
            ParallelDataHandling.log(*args, level=level)

    @property
    def is_root(self):
        return wlb.mpi.worldRank() == 0

    @property
    def world_rank(self):
        return wlb.mpi.worldRank()

    def save_all(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        if os.path.isfile(directory):
            raise RuntimeError(f"Trying to save to {directory}, but file exists already")

        for field_name, data_name in self._field_name_to_cpu_data_name.items():
            self.blocks.writeBlockData(data_name, os.path.join(directory, field_name + ".dat"))

    def load_all(self, directory):
        for field_name, data_name in self._field_name_to_cpu_data_name.items():
            self.blocks.readBlockData(data_name, os.path.join(directory, field_name + ".dat"))
