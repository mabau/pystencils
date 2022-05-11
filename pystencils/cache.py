import os
from collections.abc import Hashable
from functools import partial, wraps
from itertools import chain

from functools import lru_cache as memorycache

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


def sharedmethodcache(cache_id: str):
    """Decorator for memoization of instance methods, allowing multiple methods to use the same cache.

    This decorator caches results of instance methods per instantiated object of the surrounding class.
    It allows multiple methods to use the same cache, by passing them the same `cache_id` string.
    Cached values are stored in a dictionary, which is added as a member `self.<cache_id>` to the 
    `self` object instance. Make sure that this doesn't cause any naming conflicts with other members!
    Of course, for this to be useful, said methods must have the same signature (up to additional kwargs)
    and must return the same result when called with the same arguments."""
    def _decorator(user_method):
        @wraps(user_method)
        def _decorated_func(self, *args, **kwargs):
            objdict = self.__dict__
            cache = objdict.setdefault(cache_id, dict())

            key = args
            for item in kwargs.items():
                key += item

            if key not in cache:
                result = user_method(self, *args, **kwargs)
                cache[key] = result
                return result
            else:
                return cache[key]
        return _decorated_func
    return _decorator


# Disable memory cache:
# disk_cache = lambda o: o
# disk_cache_no_fallback = lambda o: o
