from .serial_datahandling import SerialDataHandling

try:
    import waLBerla
    if waLBerla.cpp_available:
        from .parallel_datahandling import ParallelDataHandling
    else:
        waLBerla = None
except ImportError:
    waLBerla = None
    ParallelDataHandling = None


def createDataHandling(parallel, domainSize, periodicity, defaultLayout='SoA', defaultGhostLayers=1):
    if parallel:
        if waLBerla is None:
            raise ValueError("Cannot create parallel data handling because waLBerla module is not available")

        if periodicity is False or periodicity is None:
            periodicity = (0, 0, 0)
        elif periodicity is True:
            periodicity = (1, 1, 1)
        else:
            periodicity = (int(bool(x)) for x in periodicity)
            if len(periodicity) == 2:
                periodicity += (1,)

        if len(domainSize) == 2:
            dim = 2
            domainSize = (domainSize[0], domainSize[1], 1)
        else:
            dim = 3

        blockStorage = waLBerla.createUniformBlockGrid(cells=domainSize, periodicity=periodicity)
        return ParallelDataHandling(blocks=blockStorage, dim=dim,
                                    defaultLayout=defaultLayout, defaultGhostLayers=defaultGhostLayers)
    else:
        return SerialDataHandling(domainSize, periodicity=periodicity,
                                  defaultLayout=defaultLayout, defaultGhostLayers=defaultGhostLayers)
