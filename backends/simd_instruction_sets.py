

# noinspection SpellCheckingInspection
def x86_vector_instruction_set(data_type='double', instruction_set='avx'):
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

        'makeVec': 'set[0,0,0,0]',
        'makeZero': 'setzero[]',

        'loadU': 'loadu[0]',
        'loadA': 'load[0]',
        'storeU': 'storeu[0,1]',
        'storeA': 'store [0,1]',
    }

    headers = {
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

    result = {}
    pre = prefix[instruction_set]
    suf = suffix[data_type]
    for intrinsic_id, function_shortcut in base_names.items():
        function_shortcut = function_shortcut.strip()
        name = function_shortcut[:function_shortcut.index('[')]
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

    result['width'] = width[(data_type, instruction_set)]
    result['dataTypePrefix'] = {
        'double': "_" + pre + 'd',
        'float': "_" + pre,
    }

    bit_width = result['width'] * 64
    result['double'] = "__m%dd" % (bit_width,)
    result['float'] = "__m%d" % (bit_width,)
    result['int'] = "__m%di" % (bit_width,)
    result['bool'] = "__m%dd" % (bit_width,)

    result['headers'] = headers[instruction_set]
    return result


selected_instruction_set = {
    'float': x86_vector_instruction_set('float', 'avx'),
    'double': x86_vector_instruction_set('double', 'avx'),
}
