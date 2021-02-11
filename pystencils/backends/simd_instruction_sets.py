from pystencils.backends.x86_instruction_sets import get_vector_instruction_set_x86
from pystencils.backends.arm_instruction_sets import get_vector_instruction_set_arm


def get_vector_instruction_set(data_type='double', instruction_set='avx', q_registers=True):
    if instruction_set in ['neon', 'sve']:
        return get_vector_instruction_set_arm(data_type, instruction_set, q_registers)
    else:
        return get_vector_instruction_set_x86(data_type, instruction_set)


def get_supported_instruction_sets():
    """List of supported instruction sets on current hardware, or None if query failed."""
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
