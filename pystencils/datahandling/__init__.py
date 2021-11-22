import warnings

from typing import Tuple, Union

from .datahandling_interface import DataHandling
from ..enums import Target
from .serial_datahandling import SerialDataHandling

try:
    # noinspection PyPep8Naming
    import waLBerla as wlb
    if wlb.cpp_available:
        from pystencils.datahandling.parallel_datahandling import ParallelDataHandling
    else:
        wlb = None
except ImportError:
    wlb = None
    ParallelDataHandling = None


def create_data_handling(domain_size: Tuple[int, ...],
                         periodicity: Union[bool, Tuple[bool, ...]] = False,
                         default_layout: str = 'SoA',
                         default_target: Target = Target.CPU,
                         parallel: bool = False,
                         default_ghost_layers: int = 1) -> DataHandling:
    """Creates a data handling instance.

    Args:
        domain_size: size of the rectangular domain
        periodicity: either True, False for full or no periodicity or a tuple of booleans indicating periodicity
                     for each coordinate
        default_layout: default array layout, that is used if not explicitly specified in 'add_array'
        default_target: `Target`
        parallel: if True a parallel domain is created using walberla - each MPI process gets a part of the domain
        default_ghost_layers: default number of ghost layers if not overwritten in 'add_array'
    """
    if isinstance(default_target, str):
        new_target = Target[default_target.upper()]
        warnings.warn(f'Target "{default_target}" as str is deprecated. Use {new_target} instead',
                      category=DeprecationWarning)
        default_target = new_target

    if parallel:
        if wlb is None:
            raise ValueError("Cannot create parallel data handling because walberla module is not available")

        if periodicity is False or periodicity is None:
            periodicity = (0, 0, 0)
        elif periodicity is True:
            periodicity = (1, 1, 1)
        else:
            periodicity = tuple(int(bool(x)) for x in periodicity)
            if len(periodicity) == 2:
                periodicity += (1,)

        if len(domain_size) == 2:
            dim = 2
            domain_size = (domain_size[0], domain_size[1], 1)
        else:
            dim = 3

        # noinspection PyArgumentList
        block_storage = wlb.createUniformBlockGrid(cells=domain_size, periodic=periodicity)
        return ParallelDataHandling(blocks=block_storage, dim=dim, default_target=default_target,
                                    default_layout=default_layout, default_ghost_layers=default_ghost_layers)
    else:
        return SerialDataHandling(domain_size,
                                  periodicity=periodicity,
                                  default_target=default_target,
                                  default_layout=default_layout,
                                  default_ghost_layers=default_ghost_layers)


__all__ = ['create_data_handling']
