import os
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
from typing import Mapping
from collections import Counter


class DotDict(dict):
    """Normal dict with additional dot access for all keys"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def recursive_dict_update(d, u):
    """Updates the first dict argument, using second dictionary recursively.

    Examples:
        >>> d = {'sub_dict': {'a': 1, 'b': 2}, 'outer': 42}
        >>> u = {'sub_dict': {'a': 5, 'c': 10}, 'outer': 41, 'outer2': 43}
        >>> recursive_dict_update(d, u)
        {'sub_dict': {'a': 5, 'b': 2, 'c': 10}, 'outer': 41, 'outer2': 43}
    """
    d = d.copy()
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = recursive_dict_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


@contextmanager
def file_handle_for_atomic_write(file_path):
    """Open temporary file object that atomically moves to destination upon exiting.

    Allows reading and writing to and from the same filename.
    The file will not be moved to destination in case of an exception.

    Args:
        file_path: path to file to be opened
    """
    target_folder = os.path.dirname(os.path.abspath(file_path))
    with NamedTemporaryFile(delete=False, dir=target_folder, mode='w') as f:
        try:
            yield f
        finally:
            f.flush()
            os.fsync(f.fileno())
    os.rename(f.name, file_path)


@contextmanager
def atomic_file_write(file_path):
    target_folder = os.path.dirname(os.path.abspath(file_path))
    with NamedTemporaryFile(delete=False, dir=target_folder) as f:
        f.file.close()
        yield f.name
    os.rename(f.name, file_path)


def fully_contains(l1, l2):
    """Tests if elements of sequence 1 are in sequence 2 in same or higher number.

    >>> fully_contains([1, 1, 2], [1, 2])  # 1 is only present once in second list
    False
    >>> fully_contains([1, 1, 2], [1, 1, 4, 2])
    True
    """
    l1_counter = Counter(l1)
    l2_counter = Counter(l2)
    for element, count in l1_counter.items():
        if l2_counter[element] < count:
            return False
    return True
