import math
import os
import platform
from ctypes import CDLL
from warnings import warn

import numpy as np

from pystencils.backends.x86_instruction_sets import get_vector_instruction_set_x86
from pystencils.backends.arm_instruction_sets import get_vector_instruction_set_arm
from pystencils.backends.ppc_instruction_sets import get_vector_instruction_set_ppc
from pystencils.backends.riscv_instruction_sets import get_vector_instruction_set_riscv
from pystencils.typing import numpy_name_to_c


def get_vector_instruction_set(data_type='double', instruction_set='avx'):
    if data_type == 'float':
        warn(f"Ambiguous input for data_type: {data_type}. For single precision please use float32. "
             f"For more information please take numpy.dtype as a reference. This input will not be supported in future "
             f"releases")
        data_type = 'float64'

    type_name = numpy_name_to_c(np.dtype(data_type).name)

    if instruction_set in ['neon'] or instruction_set.startswith('sve'):
        return get_vector_instruction_set_arm(type_name, instruction_set)
    elif instruction_set in ['vsx']:
        return get_vector_instruction_set_ppc(type_name, instruction_set)
    elif instruction_set in ['rvv']:
        return get_vector_instruction_set_riscv(type_name, instruction_set)
    else:
        return get_vector_instruction_set_x86(type_name, instruction_set)


_cache = None
_cachelinesize = None


def get_supported_instruction_sets():
    """List of supported instruction sets on current hardware, or None if query failed."""
    global _cache
    if _cache is not None:
        return _cache.copy()
    if 'PYSTENCILS_SIMD' in os.environ:
        return os.environ['PYSTENCILS_SIMD'].split(',')
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':  # not supported by cpuinfo
        return ['neon']
    elif platform.system() == 'Linux' and platform.machine().startswith('riscv'):  # not supported by cpuinfo
        libc = CDLL('libc.so.6')
        hwcap = libc.getauxval(16)  # AT_HWCAP
        hwcap_isa_v = 1 << (ord('V') - ord('A'))  # COMPAT_HWCAP_ISA_V
        return ['rvv'] if hwcap & hwcap_isa_v else []
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
    required_sve_flags = {'sve'}
    flags = set(get_cpu_info()['flags'])
    if flags.issuperset(required_sse_flags):
        result.append("sse")
    if flags.issuperset(required_avx_flags):
        result.append("avx")
    if flags.issuperset(required_avx512_flags):
        result.append("avx512")
    if flags.issuperset(required_neon_flags):
        result.append("neon")
    if flags.issuperset(required_sve_flags):
        if platform.system() == 'Linux':
            libc = CDLL('libc.so.6')
            native_length = 8 * libc.prctl(51, 0, 0, 0, 0)  # PR_SVE_GET_VL
            if native_length < 0:
                raise OSError("SVE length query failed")
            pwr2_length = int(2**math.floor(math.log2(native_length)))
            if pwr2_length % 256 == 0:
                result.append(f"sve{pwr2_length//2}")
            if native_length != pwr2_length:
                result.append(f"sve{pwr2_length}")
            result.append(f"sve{native_length}")
        result.append("sve")
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
    from pystencils.astnodes import SympyAssignment
    import numpy as np
    from pystencils.cpu.vectorization import CachelineSize
    
    arr = np.zeros((1, 1), dtype=np.float32)
    f = ps.Field.create_from_numpy_array('f', arr, index_dimensions=0)
    ass = [CachelineSize(), SympyAssignment(f.center, CachelineSize.symbol)]
    ast = ps.create_kernel(ass, cpu_vectorize_info={'instruction_set': instruction_set})
    kernel = ast.compile()
    kernel(**{f.name: arr, CachelineSize.symbol.name: 0})
    _cachelinesize = int(arr[0, 0])
    return _cachelinesize
