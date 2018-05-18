

# noinspection SpellCheckingInspection
def get_vector_instruction_set(data_type='double', instruction_set='avx'):
    base_names = {
        '+': 'add[0, 1]',
        '-': 'sub[0, 1]',
        '*': 'mul[0, 1]',
        '/': 'div[0, 1]',

        '==': 'cmp[0, 1, _CMP_EQ_UQ  ]',
        '!=': 'cmp[0, 1, _CMP_NEQ_UQ ]',
        '>=': 'cmp[0, 1, _CMP_GE_OQ  ]',
        '<=': 'cmp[0, 1, _CMP_LE_OQ  ]',
        '<': 'cmp[0, 1, _CMP_NGE_UQ ]',
        '>': 'cmp[0, 1, _CMP_NLE_UQ ]',
        '&': 'and[0, 1]',
        '|': 'or[0, 1]',
        'blendv': 'blendv[0, 1, 2]',

        'sqrt': 'sqrt[0]',

        'makeVec': 'set[]',
        'makeZero': 'setzero[]',

        'loadU': 'loadu[0]',
        'loadA': 'load[0]',
        'storeU': 'storeu[0,1]',
        'storeA': 'store[0,1]',
        'stream': 'stream[0,1]',
    }

    headers = {
        'avx512': ['<immintrin.h>'],
        'avx': ['<immintrin.h>'],
        'sse': ['<xmmintrin.h>', '<emmintrin.h>', '<pmmintrin.h>', '<tmmintrin.h>', '<smmintrin.h>', '<nmmintrin.h>']
    }

    suffix = {
        'double': 'pd',
        'float': 'ps',
    }
    prefix = {
        'sse': '_mm',
        'avx': '_mm256',
        'avx512': '_mm512',
    }

    width = {
        ("double", "sse"): 2,
        ("float", "sse"): 4,
        ("double", "avx"): 4,
        ("float", "avx"): 8,
        ("double", "avx512"): 8,
        ("float", "avx512"): 16,
    }

    result = {
        'width': width[(data_type, instruction_set)],
    }
    pre = prefix[instruction_set]
    suf = suffix[data_type]
    for intrinsic_id, function_shortcut in base_names.items():
        function_shortcut = function_shortcut.strip()
        name = function_shortcut[:function_shortcut.index('[')]

        if intrinsic_id == 'makeVec':
            arg_string = "({})".format(",".join(["{0}"] * result['width']))
        else:
            args = function_shortcut[function_shortcut.index('[') + 1: -1]
            arg_string = "("
            for arg in args.split(","):
                arg = arg.strip()
                if not arg:
                    continue
                if arg in ('0', '1', '2', '3', '4', '5'):
                    arg_string += "{" + arg + "},"
                else:
                    arg_string += arg + ","
            arg_string = arg_string[:-1] + ")"
        result[intrinsic_id] = pre + "_" + name + "_" + suf + arg_string

    result['dataTypePrefix'] = {
        'double': "_" + pre + 'd',
        'float': "_" + pre,
    }

    bit_width = result['width'] * (64 if data_type == 'double' else 32)
    result['double'] = "__m%dd" % (bit_width,)
    result['float'] = "__m%d" % (bit_width,)
    result['int'] = "__m%di" % (bit_width,)
    result['bool'] = "__m%dd" % (bit_width,)

    result['headers'] = headers[instruction_set]
    return result


def get_supported_instruction_sets():
    """List of supported instruction sets on current hardware, or None if query failed."""
    try:
        from cpuinfo import get_cpu_info
    except ImportError:
        return None

    result = []
    required_sse_flags = {'sse', 'sse2', 'ssse3', 'sse4_1', 'sse4_2'}
    required_avx_flags = {'avx'}
    required_avx512_flags = {'avx512f'}
    flags = set(get_cpu_info()['flags'])
    if flags.issuperset(required_sse_flags):
        result.append("sse")
    if flags.issuperset(required_avx_flags):
        result.append("avx")
    if flags.issuperset(required_avx512_flags):
        result.append("avx512")
    return result
