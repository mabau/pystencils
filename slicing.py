import sympy as sp


class SliceMaker(object):
    def __getitem__(self, item):
        return item
makeSlice = SliceMaker()


def normalizeSlice(slices, sizes):
    """Converts slices with floating point and/or negative entries to integer slices"""

    if len(slices) != len(sizes):
        raise ValueError("Slice dimension does not match sizes")

    result = []

    for s, size in zip(slices, sizes):
        if type(s) is int:
            result.append(s)
            continue
        if type(s) is float:
            result.append(int(s * size))
            continue

        assert (type(s) is slice)

        if s.start is None:
            newStart = 0
        elif type(s.start) is float:
            newStart = int(s.start * size)
        else:
            newStart = s.start

        if s.stop is None:
            newStop = size
        elif type(s.stop) is float:
            newStop = int(s.stop * size)
        elif not isinstance(s.stop, sp.Basic) and s.stop < 0:
            newStop = size + s.stop
        else:
            newStop = s.stop

        result.append(slice(newStart, newStop, s.step if s.step is not None else 1))

    return tuple(result)


def sliceFromDirection(directionName, dim, normalOffset=0, tangentialOffset=0):
    """
    Create a slice from a direction named by compass scheme:
        i.e. 'N' for north returns same as makeSlice[:, -1]
        the naming is:
            - x: W, E (west, east)
            - y: S, N (south, north)
            - z: B, T (bottom, top)
    Also combinations are allowed like north-east 'NE'

    :param directionName: name of direction as explained above
    :param dim: dimension of the returned slice (should be 2 or 3)
    :param normalOffset: the offset in 'normal' direction: e.g. sliceFromDirection('N',2, normalOffset=2)
                         would return makeSlice[:, -3]
    :param tangentialOffset: offset in the other directions: e.g. sliceFromDirection('N',2, tangentialOffset=2)
                         would return makeSlice[2:-2, -1]
    """
    if tangentialOffset == 0:
        result = [slice(None, None, None)] * dim
    else:
        result = [slice(tangentialOffset, -tangentialOffset, None)] * dim

    normalSliceHigh, normalSliceLow = -1-normalOffset, normalOffset

    for dimIdx, (lowName, highName) in enumerate([('W', 'E'), ('S', 'N'), ('B', 'T')]):
        if lowName in directionName:
            assert highName not in directionName, "Invalid direction name"
            result[dimIdx] = normalSliceLow
        if highName in directionName:
            assert lowName not in directionName, "Invalid direction name"
            result[dimIdx] = normalSliceHigh
    return tuple(result)
