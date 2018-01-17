import sympy as sp
import json
import os

try:
    from functools import lru_cache as memorycache
except ImportError:
    from backports.functools_lru_cache import lru_cache as memorycache

try:
    from joblib import Memory
    from appdirs import user_cache_dir
    if 'PYSTENCILS_CACHE_DIR' in os.environ:
        cacheDir = os.environ['PYSTENCILS_CACHE_DIR']
    else:
        cacheDir = user_cache_dir('pystencils')
    diskcache = Memory(cachedir=cacheDir, verbose=False).cache
    diskcacheNoFallback = diskcache
except ImportError:
    # fallback to in-memory caching if joblib is not available
    diskcache = memorycache(maxsize=64)
    diskcacheNoFallback = lambda o: o


# Disable memory cache:
# diskcache = lambda o: o
# diskcacheNoFallback = lambda o: o
