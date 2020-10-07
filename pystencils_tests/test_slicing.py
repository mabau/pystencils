import numpy as np
from numpy.testing import assert_array_equal
from pystencils import create_data_handling
from pystencils.slicing import SlicedGetter, make_slice, SlicedGetterDataHandling, shift_slice, slice_intersection


def test_sliced_getter():
    def get_slice(slice_obj=None):
        arr = np.ones((10, 10))
        if slice_obj is None:
            slice_obj = make_slice[:, :]

        return arr[slice_obj]

    sli = SlicedGetter(get_slice)

    test = make_slice[2:-2, 2:-2]
    assert sli[test].shape == (6, 6)


def test_sliced_getter_data_handling():
    domain_shape = (10, 10)

    dh = create_data_handling(domain_size=domain_shape, default_ghost_layers=1)
    dh.add_array("src", values_per_cell=1)
    dh.fill("src", 1.0, ghost_layers=True)

    dh.add_array("dst", values_per_cell=1)
    dh.fill("dst", 0.0, ghost_layers=True)

    sli = SlicedGetterDataHandling(dh, 'dst')
    slice_obj = make_slice[2:-2, 2:-2]
    assert np.sum(sli[slice_obj]) == 0

    sli = SlicedGetterDataHandling(dh, 'src')
    slice_obj = make_slice[2:-2, 2:-2]
    assert np.sum(sli[slice_obj]) == 36


def test_shift_slice():

    sh = shift_slice(make_slice[2:-2, 2:-2], [1, 2])
    assert sh[0] == slice(3, -1, None)
    assert sh[1] == slice(4, 0, None)

    sh = shift_slice(make_slice[2:-2, 2:-2], 1)
    assert sh[0] == slice(3, -1, None)
    assert sh[1] == slice(3, -1, None)

    sh = shift_slice([2, 4], 1)
    assert sh[0] == 3
    assert sh[1] == 5

    sh = shift_slice([2, None], 1)
    assert sh[0] == 3
    assert sh[1] is None

    sh = shift_slice([1.5, 1.5], 1)
    assert sh[0] == 1.5
    assert sh[1] == 1.5


def test_shifted_array_access():
    arr = np.array(range(10))
    
    sh = make_slice[2:5]
    assert_array_equal(arr[sh], [2,3,4])

    sh = shift_slice(sh, 3)
    assert_array_equal(arr[sh], [5,6,7])

    arr = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    sh = make_slice[0:2, 0:2]
    assert_array_equal(arr[sh], [[1, 2], [4, 5]])

    sh = shift_slice(sh, (1,1))
    assert_array_equal(arr[sh], [[5, 6], [8, 9]])


def test_slice_intersection():
    sl1 = make_slice[1:10, 1:10]
    sl2 = make_slice[5:15, 5:15]

    intersection = slice_intersection(sl1, sl2)
    assert intersection[0] == slice(5, 10, None)
    assert intersection[1] == slice(5, 10, None)

    sl2 = make_slice[12:15, 12:15]

    intersection = slice_intersection(sl1, sl2)
    assert intersection is None
