import os

try:
    from functools import lru_cache as memorycache
except ImportError:
    from backports.functools_lru_cache import lru_cache as memorycache

try:
    from joblib import Memory
    from appdirs import user_cache_dir
    if 'PYSTENCILS_CACHE_DIR' in os.environ:
        cache_dir = os.environ['PYSTENCILS_CACHE_DIR']
    else:
        cache_dir = user_cache_dir('pystencils')
    disk_cache = Memory(cachedir=cache_dir, verbose=False).cache
    disk_cache_no_fallback = disk_cache
except ImportError:
    # fallback to in-memory caching if joblib is not available
    disk_cache = memorycache(maxsize=64)

    def disk_cache_no_fallback(o):
        return o


# Disable memory cache:
# disk_cache = lambda o: o
# disk_cache_no_fallback = lambda o: o
