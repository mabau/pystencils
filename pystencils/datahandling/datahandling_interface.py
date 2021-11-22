from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from pystencils.enums import Target, Backend
from pystencils.field import Field, FieldType


class DataHandling(ABC):
    """
    Manages the storage of arrays and maps them to a symbolic field.
    Two versions are available: a simple, pure Python implementation for single node
    simulations :py:class:SerialDataHandling and a distributed version using walberla in :py:class:ParallelDataHandling

    Keep in mind that the data can be distributed, so use the 'access' method whenever possible and avoid the
    'gather' function that has collects (parts of the) distributed data on a single process.
    """

    _GPU_LIKE_TARGETS = [Target.GPU]
    _GPU_LIKE_BACKENDS = [Backend.CUDA]

    # ---------------------------- Adding and accessing data -----------------------------------------------------------
    @property
    @abstractmethod
    def default_target(self) -> Target:
        """Target Enum indicating the target of the computation"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the domain, either 2 or 3"""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Shape of outer bounding box."""

    @property
    @abstractmethod
    def periodicity(self) -> Tuple[bool, ...]:
        """Returns tuple of booleans for x,y,(z) directions with True if domain is periodic in that direction."""

    @abstractmethod
    def add_array(self, name: str, values_per_cell, dtype=np.float64,
                  latex_name: Optional[str] = None, ghost_layers: Optional[int] = None, layout: Optional[str] = None,
                  cpu: bool = True, gpu: Optional[bool] = None, alignment=False, field_type=FieldType.GENERIC) -> Field:
        """Adds a (possibly distributed) array to the handling that can be accessed using the given name.

        For each array a symbolic field is available via the 'fields' dictionary

        Args:
            name: unique name that is used to access the field later
            values_per_cell: shape of the dim+1 coordinate. DataHandling supports zero or one index dimensions,
                             i.e. scalar fields and vector fields. This parameter gives the shape of the index
                             dimensions. The default value of 1 means no index dimension are created.
            dtype: data type of the array as numpy data type
            latex_name: optional, name of the symbolic field, if not given 'name' is used
            ghost_layers: number of ghost layers - if not specified a default value specified in the constructor
                         is used
            layout: memory layout of array, either structure of arrays 'SoA' or array of structures 'AoS'.
                    this is only important if values_per_cell > 1
            cpu: allocate field on the CPU
            gpu: allocate field on the GPU, if None, a GPU field is allocated if default_target is 'GPU'
            alignment: either False for no alignment, or the number of bytes to align to
        Returns:
            pystencils field, that can be used to formulate symbolic kernels
        """

    def add_arrays(self,
                   description: str,
                   dtype=np.float64,
                   ghost_layers: Optional[int] = None,
                   layout: Optional[str] = None,
                   cpu: bool = True,
                   gpu: Optional[bool] = None,
                   alignment=False,
                   field_type=FieldType.GENERIC) -> Tuple[Field]:
        """Adds multiple arrays using a string description similar to :func:`pystencils.fields`


        >>> from pystencils.datahandling import create_data_handling
        >>> dh = create_data_handling((20, 30))
        >>> x, y =dh.add_arrays('x, y(9)')
        >>> print(dh.fields)
        {'x': x: double[22,32], 'y': y(9): double[22,32]}
        >>> assert x == dh.fields['x']
        >>> assert dh.fields['x'].shape == (22, 32)
        >>> assert dh.fields['y'].index_shape == (9,)

        Args:
            description (str): String description of the fields to add
            dtype: data type of the array as numpy data type
            ghost_layers: number of ghost layers - if not specified a default value specified in the constructor
                         is used
            layout: memory layout of array, either structure of arrays 'SoA' or array of structures 'AoS'.
                    this is only important if values_per_cell > 1
            cpu: allocate field on the CPU
            gpu: allocate field on the GPU, if None, a GPU field is allocated if default_target is 'GPU'
            alignment: either False for no alignment, or the number of bytes to align to
        Returns:
            Fields representing the just created arrays
        """
        from pystencils.field import _parse_part1

        names = []
        for name, indices in _parse_part1(description):
            names.append(name)
            self.add_array(name,
                           values_per_cell=indices,
                           dtype=dtype,
                           ghost_layers=ghost_layers,
                           layout=layout,
                           cpu=cpu,
                           gpu=gpu,
                           alignment=alignment,
                           field_type=field_type)

        return (self.fields[n] for n in names)

    @abstractmethod
    def has_data(self, name):
        """Returns true if a field or custom data element with this name was added."""

    @abstractmethod
    def add_array_like(self, name, name_of_template_field, latex_name=None, cpu=True, gpu=None):
        """
        Adds an array with the same parameters (number of ghost layers, values_per_cell, dtype) as existing array.

        Args:
            name: name of new array
            name_of_template_field: name of array that is used as template
            latex_name: see 'add' method
            cpu: see 'add' method
            gpu: see 'add' method
        """

    @abstractmethod
    def add_custom_data(self, name: str, cpu_creation_function,
                        gpu_creation_function=None, cpu_to_gpu_transfer_func=None, gpu_to_cpu_transfer_func=None):
        """Adds custom (non-array) data to domain.

        Args:
            name: name to access data
            cpu_creation_function: function returning a new instance of the data that should be stored
            gpu_creation_function: optional, function returning a new instance, stored on GPU
            cpu_to_gpu_transfer_func: function that transfers cpu to gpu version,
                                      getting two parameters (gpu_instance, cpu_instance)
            gpu_to_cpu_transfer_func: function that transfers gpu to cpu version, getting two parameters
                                      (gpu_instance, cpu_instance)
        """

    def add_custom_class(self, name: str, class_obj, cpu: bool = True, gpu: bool = False):
        """Adds non-array data by passing a class object with optional 'to_gpu' and 'to_cpu' member functions."""
        cpu_to_gpu_transfer_func = class_obj.to_gpu if cpu and gpu and hasattr(class_obj, 'to_gpu') else None
        gpu_to_cpu_transfer_func = class_obj.to_cpu if cpu and gpu and hasattr(class_obj, 'to_cpu') else None
        self.add_custom_data(name,
                             cpu_creation_function=class_obj if cpu else None,
                             gpu_creation_function=class_obj if gpu else None,
                             cpu_to_gpu_transfer_func=cpu_to_gpu_transfer_func,
                             gpu_to_cpu_transfer_func=gpu_to_cpu_transfer_func)

    @property
    @abstractmethod
    def fields(self) -> Dict[str, Field]:
        """Dictionary mapping data name to symbolic pystencils field - use this to create pystencils kernels."""

    @property
    @abstractmethod
    def array_names(self) -> Sequence[str]:
        """Sequence of all array names."""

    @property
    @abstractmethod
    def custom_data_names(self) -> Sequence[str]:
        """Sequence of all custom data names."""

    @abstractmethod
    def ghost_layers_of_field(self, name: str) -> int:
        """Returns the number of ghost layers for a specific field/array."""

    @abstractmethod
    def values_per_cell(self, name: str) -> Tuple[int, ...]:
        """Returns values_per_cell of array."""

    @abstractmethod
    def iterate(self, slice_obj=None, gpu=False, ghost_layers=None,
                inner_ghost_layers=True) -> Iterable['Block']:
        """Iterate over local part of potentially distributed data structure."""

    @abstractmethod
    def gather_array(self, name, slice_obj=None, all_gather=False, ghost_layers=False) -> Optional[np.ndarray]:
        """
        Gathers part of the domain on a local process. Whenever possible use 'access' instead, since this method copies
        the distributed data to a single process which is inefficient and may exhaust the available memory

        Args:
            name: name of the array to gather
            slice_obj: slice expression of the rectangular sub-part that should be gathered
            all_gather: if False only the root process receives the result, if True all processes
            ghost_layers: number of outer ghost layers to include (only available for serial version of data handling)

        Returns:
            gathered field that does not include any ghost layers, or None if gathered on another process
        """

    @abstractmethod
    def run_kernel(self, kernel_function, *args, **kwargs) -> None:
        """Runs a compiled pystencils kernel.

        Uses the arrays stored in the DataHandling class for all array parameters. Additional passed arguments are
        directly passed to the kernel function and override possible parameters from the DataHandling
        """

    @abstractmethod
    def get_kernel_kwargs(self, kernel_function, **kwargs):
        """Returns the input arguments of a kernel"""

    @abstractmethod
    def swap(self, name1, name2, gpu=False):
        """Swaps data of two arrays"""

    # ------------------------------- CPU/GPU transfer -----------------------------------------------------------------

    @abstractmethod
    def to_cpu(self, name):
        """Copies GPU data of array with specified name to CPU.
        Works only if 'cpu=True' and 'gpu=True' has been used in 'add' method."""

    @abstractmethod
    def to_gpu(self, name):
        """Copies GPU data of array with specified name to GPU.
        Works only if 'cpu=True' and 'gpu=True' has been used in 'add' method."""

    @abstractmethod
    def all_to_cpu(self):
        """Copies data from GPU to CPU for all arrays that have a CPU and a GPU representation."""

    @abstractmethod
    def all_to_gpu(self):
        """Copies data from CPU to GPU for all arrays that have a CPU and a GPU representation."""

    @abstractmethod
    def is_on_gpu(self, name):
        """Checks if this data was also allocated on the GPU - does not check if this data item is in synced."""

    @abstractmethod
    def create_vtk_writer(self, file_name, data_names, ghost_layers=False) -> Callable[[int], None]:
        """VTK output for one or multiple arrays.

        Args
            file_name: base file name without extension for the VTK output
            data_names: list of array names that should be included in the vtk output
            ghost_layers: true if ghost layer information should be written out as well

        Returns:
            a function that can be called with an integer time step to write the current state
            i.e create_vtk_writer('some_file', ['velocity', 'density']) (1)
        """

    @abstractmethod
    def create_vtk_writer_for_flag_array(self, file_name, data_name, masks_to_name,
                                         ghost_layers=False) -> Callable[[int], None]:
        """VTK output for an unsigned integer field, where bits are interpreted as flags.

        Args:
            file_name: see create_vtk_writer
            data_name: name of an array with uint type
            masks_to_name: dictionary mapping integer masks to a name in the output
            ghost_layers: see create_vtk_writer

        Returns:
            functor that can be called with time step
         """

    # ------------------------------- Communication --------------------------------------------------------------------

    @abstractmethod
    def synchronization_function(self, names, stencil=None, target=None, **kwargs) -> Callable[[], None]:
        """Synchronizes ghost layers for distributed arrays.

        For serial scenario this has to be called for correct periodicity handling

        Args:
            names: what data to synchronize: name of array or sequence of names
            stencil: stencil as string defining which neighbors are synchronized e.g. 'D2Q9', 'D3Q19'
                     if None, a full synchronization (i.e. D2Q9 or D3Q27) is done
            target: `Target` either 'CPU' or 'GPU'
            kwargs: implementation specific, optional optimization parameters for communication

        Returns:
            function object to run the communication
        """

    def reduce_float_sequence(self, sequence, operation, all_reduce=False) -> np.array:
        """Takes a sequence of floating point values on each process and reduces it element-wise.

        If all_reduce, all processes get the result, otherwise only the root process.
        Possible operations are 'sum', 'min', 'max'
        """

    def reduce_int_sequence(self, sequence, operation, all_reduce=False) -> np.array:
        """See function reduce_float_sequence - this is the same for integers"""

    # ------------------------------- Data access and modification -----------------------------------------------------

    def fill(self, array_name: str, val, value_idx: Optional[Union[int, Tuple[int, ...]]] = None,
             slice_obj=None, ghost_layers=False, inner_ghost_layers=False) -> None:
        """Sets all cells to the same value.

        Args:
            array_name: name of the array that should be modified
            val: value to set the array to
            value_idx: If an array stores multiple values per cell, this index chooses which of this values to fill.
                       If None, all values are set
            slice_obj: if passed, only the defined slice is filled
            ghost_layers: True if the outer ghost layers should also be filled
            inner_ghost_layers: True if the inner ghost layers should be filled. Inner ghost layers occur only in
                                parallel setups for distributed memory communication.
        """
        if ghost_layers is True:
            ghost_layers = self.ghost_layers_of_field(array_name)
        if inner_ghost_layers is True:
            ghost_layers = self.ghost_layers_of_field(array_name)

        for b in self.iterate(slice_obj, ghost_layers=ghost_layers, inner_ghost_layers=inner_ghost_layers):
            if value_idx is not None:
                if isinstance(value_idx, int):
                    value_idx = (value_idx,)
                assert len(value_idx) == len(self.values_per_cell(array_name))
                b[array_name][(Ellipsis, *value_idx)].fill(val)
            else:
                b[array_name].fill(val)

    def min(self, array_name, slice_obj=None, ghost_layers=False, inner_ghost_layers=False, reduce=True):
        """Returns the minimum value inside the domain or slice of the domain.

        For meaning of arguments see documentation of :func:`DataHandling.fill`.

        Returns:
            the minimum of the locally stored domain part is returned if reduce is False, otherwise the global minimum
            on the root process, on other processes None
        """
        result = None
        if ghost_layers is True:
            ghost_layers = self.ghost_layers_of_field(array_name)
        if inner_ghost_layers is True:
            ghost_layers = self.ghost_layers_of_field(array_name)
        for b in self.iterate(slice_obj, ghost_layers=ghost_layers, inner_ghost_layers=inner_ghost_layers):
            m = np.min(b[array_name])
            result = m if result is None else np.min(result, m)
        return self.reduce_float_sequence([result], 'min', all_reduce=True)[0] if reduce else result

    def max(self, array_name, slice_obj=None, ghost_layers=False, inner_ghost_layers=False, reduce=True):
        """Returns the maximum value inside the domain or slice of the domain.

        For argument description see :func:`DataHandling.min`
        """
        result = None
        if ghost_layers is True:
            ghost_layers = self.ghost_layers_of_field(array_name)
        if inner_ghost_layers is True:
            ghost_layers = self.ghost_layers_of_field(array_name)
        for b in self.iterate(slice_obj, ghost_layers=ghost_layers, inner_ghost_layers=inner_ghost_layers):
            m = np.max(b[array_name])
            result = m if result is None else np.max(result, m)

        return self.reduce_float_sequence([result], 'max', all_reduce=True)[0] if reduce else result

    def save_all(self, file):
        """Saves all field data to disk into a file"""

    def load_all(self, file):
        """Loads all field data from disk into a file

        Works only if save_all was called with exactly the same field sizes, layouts etc.
        When run in parallel save and load has to be called with the same number of processes.
        Use for check pointing only - to store results use VTK output
        """

    def __str__(self):
        result = ""

        first_column_width = max(len("Name"), max(len(a) for a in self.array_names))
        row_format = "{:>%d}|{:>21}|{:>21}\n" % (first_column_width,)
        separator_line = "-" * (first_column_width + 21 + 21 + 2) + "\n"
        result += row_format.format("Name", "Inner (min/max)", "WithGl (min/max)")
        result += separator_line
        for arr_name in sorted(self.array_names):
            inner_min_max = (self.min(arr_name, ghost_layers=False), self.max(arr_name, ghost_layers=False))
            with_gl_min_max = (self.min(arr_name, ghost_layers=True), self.max(arr_name, ghost_layers=True))
            inner_min_max = "({0[0]:3.3g},{0[1]:3.3g})".format(inner_min_max)
            with_gl_min_max = "({0[0]:3.3g},{0[1]:3.3g})".format(with_gl_min_max)
            result += row_format.format(arr_name, inner_min_max, with_gl_min_max)
        return result

    def log(self, *args, level='INFO'):
        """Similar to print with additional information (time, rank)."""

    def log_on_root(self, *args, level='INFO'):
        """Logs only on root process. For serial setups this is equivalent to log"""

    @property
    def is_root(self):
        """Returns True for exactly one process in the simulation"""

    @property
    def world_rank(self):
        """Number of current process"""


class Block:
    """Represents locally stored part of domain.

    Instances of this class are returned by DataHandling.iterate, do not create manually!
    """

    def __init__(self, offset, local_slice):
        self._offset = offset
        self._localSlice = local_slice

    @property
    def offset(self):
        """Offset of the current block in global coordinates (where lower ghost layers have negative indices)."""
        return self._offset

    @property
    def cell_index_arrays(self):
        """Global coordinate mesh-grid of cell coordinates.

        Cell indices start at 0 at the first inner cell, lower ghost layers have negative indices
        """
        mesh_grid_params = [offset + np.arange(width, dtype=np.int32)
                            for offset, width in zip(self.offset, self.shape)]
        return np.meshgrid(*mesh_grid_params, indexing='ij', copy=False)

    @property
    def midpoint_arrays(self):
        """Global coordinate mesh-grid of cell midpoints which are shifted by 0.5 compared to cell indices."""
        mesh_grid_params = [offset + 0.5 + np.arange(width, dtype=float)
                            for offset, width in zip(self.offset, self.shape)]
        return np.meshgrid(*mesh_grid_params, indexing='ij', copy=False)

    @property
    def shape(self):
        """Shape of the fields (potentially including ghost layers)."""
        return tuple(s.stop - s.start for s in self._localSlice[:len(self._offset)])

    @property
    def global_slice(self):
        """Slice in global coordinates."""
        return tuple(slice(off, off + size) for off, size in zip(self._offset, self.shape))

    def __getitem__(self, data_name: str) -> np.ndarray:
        raise NotImplementedError()
