try:
    from functools import lru_cache as memorycache
except ImportError:
    from backports.functools_lru_cache import lru_cache as memorycache

try:
    from joblib import Memory
    diskcache = Memory(cachedir="/tmp/lbmpy", verbose=False).cache
except ImportError:
    # fallback to in-memory caching if joblib is not available
    diskcache = memorycache(maxsize=64)


# joblibs Memory decorator does not play nicely with sphinx autodoc (decorated functions do not occur in autodoc)
# -> if this script is imported by sphinx we use functools instead
import sys
calledBySphinx = 'sphinx' in sys.modules
if calledBySphinx:
    diskcache = memorycache(maxsize=64)
