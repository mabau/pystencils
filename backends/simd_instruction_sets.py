

def x86VectorInstructionSet(dataType='double', instructionSet='avx'):
    baseNames = {
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

        'makeVec':  'set[0,0,0,0]',
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
    pre = prefix[instructionSet]
    suf = suffix[dataType]
    for intrinsicId, functionShortcut in baseNames.items():
        functionShortcut = functionShortcut.strip()
        name = functionShortcut[:functionShortcut.index('[')]
        args = functionShortcut[functionShortcut.index('[') + 1: -1]
        argString = "("
        for arg in args.split(","):
            arg = arg.strip()
            if not arg:
                continue
            if arg in ('0', '1', '2', '3', '4', '5'):
                argString += "{" + arg + "},"
            else:
                argString += arg + ","
        argString = argString[:-1] + ")"
        result[intrinsicId] = pre + "_" + name + "_" + suf + argString

    result['width'] = width[(dataType, instructionSet)]
    result['dataTypePrefix'] = {
        'double': "_" + pre + 'd',
        'float': "_" + pre,
    }

    bitWidth = result['width'] * 64
    result['double'] = "__m%dd" % (bitWidth,)
    result['float'] = "__m%d" % (bitWidth,)
    result['int'] = "__m%di" % (bitWidth,)
    result['bool'] = "__m%dd" % (bitWidth,)

    result['headers'] = headers[instructionSet]
    return result


selectedInstructionSet = {
    'float': x86VectorInstructionSet('float', 'avx'),
    'double': x86VectorInstructionSet('double', 'avx'),
}
