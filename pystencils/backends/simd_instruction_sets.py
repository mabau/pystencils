import platform

from pystencils.backends.x86_instruction_sets import get_vector_instruction_set_x86
from pystencils.backends.arm_instruction_sets import get_vector_instruction_set_arm
from pystencils.backends.ppc_instruction_sets import get_vector_instruction_set_ppc


def get_vector_instruction_set(data_type='double', instruction_set='avx'):
    if instruction_set in ['neon', 'sve']:
        return get_vector_instruction_set_arm(data_type, instruction_set)
    elif instruction_set in ['vsx']:
        return get_vector_instruction_set_ppc(data_type, instruction_set)
    else:
        return get_vector_instruction_set_x86(data_type, instruction_set)


_cache = None
_cachelinesize = None


def get_supported_instruction_sets():
    """List of supported instruction sets on current hardware, or None if query failed."""
    global _cache
    if _cache is not None:
        return _cache.copy()
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':  # not supported by cpuinfo
        return ['neon']
    elif platform.machine().startswith('ppc64'):  # no flags reported by cpuinfo
        import subprocess
        import tempfile
        from pystencils.cpu.cpujit import get_compiler_config
        f = tempfile.NamedTemporaryFile(suffix='.cpp')
        command = [get_compiler_config()['command'], '-mcpu=native', '-dM', '-E', f.name]
        macros = subprocess.check_output(command, input='', text=True)
        if '#define __VSX__' in macros and '#define __ALTIVEC__' in macros:
            _cache = ['vsx']
        else:
            _cache = []
        return _cache.copy()
    try:
        from cpuinfo import get_cpu_info
    except ImportError:
        return None

    result = []
    required_sse_flags = {'sse', 'sse2', 'ssse3', 'sse4_1', 'sse4_2'}
    required_avx_flags = {'avx', 'avx2'}
    required_avx512_flags = {'avx512f'}
    required_neon_flags = {'neon'}
    flags = set(get_cpu_info()['flags'])
    if flags.issuperset(required_sse_flags):
        result.append("sse")
    if flags.issuperset(required_avx_flags):
        result.append("avx")
    if flags.issuperset(required_avx512_flags):
        result.append("avx512")
    if flags.issuperset(required_neon_flags):
        result.append("neon")
    return result


def get_cacheline_size(instruction_set):
    """Get the size (in bytes) of a cache block that can be zeroed without memory access.
       Usually, this is identical to the cache line size."""
    global _cachelinesize
    
    instruction_sets = get_vector_instruction_set('double', instruction_set)
    if 'cachelineSize' not in instruction_sets:
        return None
    if _cachelinesize is not None:
        return _cachelinesize
    
    import pystencils as ps
    import numpy as np
    
    arr = np.zeros((1, 1), dtype=np.float32)
    f = ps.Field.create_from_numpy_array('f', arr, index_dimensions=0)
    ass = [ps.astnodes.CachelineSize(), ps.Assignment(f.center, ps.astnodes.CachelineSize.symbol)]
    ast = ps.create_kernel(ass, cpu_vectorize_info={'instruction_set': instruction_set})
    kernel = ast.compile()
    kernel(**{f.name: arr, ps.astnodes.CachelineSize.symbol.name: 0})
    _cachelinesize = int(arr[0, 0])
    return _cachelinesize
