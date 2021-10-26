import os
from collections.abc import Hashable
from functools import partial
from itertools import chain

try:
    from functools import lru_cache as memorycache
except ImportError:
    from backports.functools_lru_cache import lru_cache as memorycache

from joblib import Memory
from appdirs import user_cache_dir

if 'PYSTENCILS_CACHE_DIR' in os.environ:
    cache_dir = os.environ['PYSTENCILS_CACHE_DIR']
else:
    cache_dir = user_cache_dir('pystencils')
disk_cache = Memory(cache_dir, verbose=False).cache
disk_cache_no_fallback = disk_cache


def _wrapper(wrapped_func, cached_func, *args, **kwargs):
    if all(isinstance(a, Hashable) for a in chain(args, kwargs.values())):
        return cached_func(*args, **kwargs)
    else:
        return wrapped_func(*args, **kwargs)


def memorycache_if_hashable(maxsize=128, typed=False):
    def wrapper(func):
        return partial(_wrapper, func, memorycache(maxsize, typed)(func))

    return wrapper

# Disable memory cache:
# disk_cache = lambda o: o
# disk_cache_no_fallback = lambda o: o
