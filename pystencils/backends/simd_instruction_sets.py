

# noinspection SpellCheckingInspection
def get_vector_instruction_set(data_type='double', instruction_set='avx'):
    comparisons = {
        '==': '_CMP_EQ_UQ',
        '!=': '_CMP_NEQ_UQ',
        '>=': '_CMP_GE_OQ',
        '<=': '_CMP_LE_OQ',
        '<': '_CMP_NGE_UQ',
        '>': '_CMP_NLE_UQ',
    }
    base_names = {
        '+': 'add[0, 1]',
        '-': 'sub[0, 1]',
        '*': 'mul[0, 1]',
        '/': 'div[0, 1]',
        '&': 'and[0, 1]',
        '|': 'or[0, 1]',
        'blendv': 'blendv[0, 1, 2]',

        'sqrt': 'sqrt[0]',

        'makeVecConst': 'set[]',
        'makeVec': 'set[]',
        'makeVecBool': 'set[]',
        'makeVecConstBool': 'set[]',
        'makeZero': 'setzero[]',

        'loadU': 'loadu[0]',
        'loadA': 'load[0]',
        'storeU': 'storeu[0,1]',
        'storeA': 'store[0,1]',
        'stream': 'stream[0,1]',
        'maskstore': 'mask_store[0, 2, 1]' if instruction_set == 'avx512' else 'maskstore[0, 2, 1]',
        'maskload': 'mask_load[0, 2, 1]' if instruction_set == 'avx512' else 'maskload[0, 2, 1]'
    }
    if instruction_set == 'avx512':
        base_names.update({
            'maskStore': 'mask_store[0, 2, 1]',
            'maskStoreU': 'mask_storeu[0, 2, 1]',
            'maskLoad': 'mask_load[2, 1, 0]',
            'maskLoadU': 'mask_loadu[2, 1, 0]'
        })
    if instruction_set == 'avx':
        base_names.update({
            'maskStore': 'maskstore[0, 2, 1]',
            'maskStoreU': 'maskstore[0, 2, 1]',
            'maskLoad': 'maskload[0, 1]',
            'maskLoadU': 'maskloadu[0, 1]'
        })

    for comparison_op, constant in comparisons.items():
        base_names[comparison_op] = f'cmp[0, 1, {constant}]'

    headers = {
        'avx512': ['<immintrin.h>'],
        'avx': ['<immintrin.h>'],
        'sse': ['<immintrin.h>', '<xmmintrin.h>', '<emmintrin.h>', '<pmmintrin.h>',
                '<tmmintrin.h>', '<smmintrin.h>', '<nmmintrin.h>']
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

        if intrinsic_id == 'makeVecConst':
            arg_string = f"({','.join(['{0}'] * result['width'])})"
        elif intrinsic_id == 'makeVec':
            params = ["{" + str(i) + "}" for i in reversed(range(result['width']))]
            arg_string = f"({','.join(params)})"
        elif intrinsic_id == 'makeVecBool':
            params = [f"(({{{i}}} ? -1.0 : 0.0)" for i in reversed(range(result['width']))]
            arg_string = f"({','.join(params)})"
        elif intrinsic_id == 'makeVecConstBool':
            params = ["(({0}) ? -1.0 : 0.0)" for _ in range(result['width'])]
            arg_string = f"({','.join(params)})"
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
        mask_suffix = '_mask' if instruction_set == 'avx512' and intrinsic_id in comparisons.keys() else ''
        result[intrinsic_id] = pre + "_" + name + "_" + suf + mask_suffix + arg_string

    result['dataTypePrefix'] = {
        'double': "_" + pre + 'd',
        'float': "_" + pre,
    }

    result['rsqrt'] = None
    bit_width = result['width'] * (64 if data_type == 'double' else 32)
    result['double'] = "__m%dd" % (bit_width,)
    result['float'] = "__m%d" % (bit_width,)
    result['int'] = "__m%di" % (bit_width,)
    result['bool'] = "__m%dd" % (bit_width,)

    result['headers'] = headers[instruction_set]
    result['any'] = "%s_movemask_%s({0}) > 0" % (pre, suf)
    result['all'] = "%s_movemask_%s({0}) == 0xF" % (pre, suf)

    if instruction_set == 'avx512':
        size = 8 if data_type == 'double' else 16
        result['&'] = '_kand_mask%d({0}, {1})' % (size,)
        result['|'] = '_kor_mask%d({0}, {1})' % (size,)
        result['any'] = '!_ktestz_mask%d_u8({0}, {0})' % (size, )
        result['all'] = '_kortestc_mask%d_u8({0}, {0})' % (size, )
        result['blendv'] = '%s_mask_blend_%s({2}, {0}, {1})' % (pre, suf)
        result['rsqrt'] = "_mm512_rsqrt14_%s({0})" % (suf,)
        result['bool'] = "__mmask%d" % (size,)

        params = " | ".join(["({{{i}}} ? {power} : 0)".format(i=i, power=2 ** i) for i in range(8)])
        result['makeVecBool'] = f"__mmask8(({params}) )"
        params = " | ".join(["({{0}} ? {power} : 0)".format(power=2 ** i) for i in range(8)])
        result['makeVecConstBool'] = f"__mmask8(({params}) )"

    if instruction_set == 'avx' and data_type == 'float':
        result['rsqrt'] = "_mm256_rsqrt_ps({0})"

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
