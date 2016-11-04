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
