import sympy as sp
import json

try:
    from functools import lru_cache as memorycache
except ImportError:
    from backports.functools_lru_cache import lru_cache as memorycache

try:
    from joblib import Memory
    diskcache = Memory(cachedir="/tmp/pystencils/joblib_memcache", verbose=False).cache
except ImportError:
    # fallback to in-memory caching if joblib is not available
    diskcache = memorycache(maxsize=64)


# joblibs Memory decorator does not play nicely with sphinx autodoc (decorated functions do not occur in autodoc)
# -> if this script is imported by sphinx we use functools instead
import sys
calledBySphinx = 'sphinx' in sys.modules
if calledBySphinx:
    diskcache = memorycache(maxsize=64)


# ------------------------ Helper classes to JSON serialize sympy objects ----------------------------------------------


class SympyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, sp.Basic):
            return {"_type": "sp", "str": str(obj)}
        else:
            super(SympyJSONEncoder, self).default(obj)


class SympyJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if '_type' in obj:
            return sp.sympify(obj['str'])
        else:
            return obj