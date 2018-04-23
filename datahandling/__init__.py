from typing import Tuple, Union
from .serial_datahandling import SerialDataHandling
from .datahandling_interface import DataHandling

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
                         parallel: bool = False,
                         default_ghost_layers: int = 1) -> DataHandling:
    """Creates a data handling instance.

    Args:
        parallel:
        domain_size:
        periodicity:
        default_layout:
        default_ghost_layers:

    Returns:

    """
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
        return ParallelDataHandling(blocks=block_storage, dim=dim,
                                    default_layout=default_layout, default_ghost_layers=default_ghost_layers)
    else:
        return SerialDataHandling(domain_size, periodicity=periodicity,
                                  default_layout=default_layout, default_ghost_layers=default_ghost_layers)


__all__ = ['create_data_handling']
