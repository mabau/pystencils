def get_argument_string(intrinsic_id, width, function_shortcut):
    if intrinsic_id == 'makeVecConst' or intrinsic_id == 'makeVecConstInt':
        arg_string = f"({','.join(['{0}'] * width)})"
    elif intrinsic_id == 'makeVec' or intrinsic_id == 'makeVecInt':
        params = ["{" + str(i) + "}" for i in reversed(range(width))]
        arg_string = f"({','.join(params)})"
    elif intrinsic_id == 'makeVecBool':
        params = [f"(({{{i}}} ? -1.0 : 0.0)" for i in reversed(range(width))]
        arg_string = f"({','.join(params)})"
    elif intrinsic_id == 'makeVecConstBool':
        params = ["(({0}) ? -1.0 : 0.0)" for _ in range(width)]
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
    return arg_string


def get_vector_instruction_set_x86(data_type='double', instruction_set='avx'):
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
        'makeVecInt': 'set[]',
        'makeVecConstInt': 'set[]',

        'loadU': 'loadu[0]',
        'loadA': 'load[0]',
        'storeU': 'storeu[0,1]',
        'storeA': 'store[0,1]',
        'stream': 'stream[0,1]',
        'maskStoreA': 'mask_store[0, 2, 1]' if instruction_set == 'avx512' else 'maskstore[0, 2, 1]',
        'maskStoreU': 'mask_storeu[0, 2, 1]' if instruction_set == 'avx512' else 'maskstore[0, 2, 1]',
    }

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
        'int': 'epi32'
    }
    prefix = {
        'sse': '_mm',
        'avx': '_mm256',
        'avx512': '_mm512',
    }

    width = {
        ("double", "sse"): 2,
        ("float", "sse"): 4,
        ("int", "sse"): 4,
        ("double", "avx"): 4,
        ("float", "avx"): 8,
        ("int", "avx"): 8,
        ("double", "avx512"): 8,
        ("float", "avx512"): 16,
        ("int", "avx512"): 16,
    }
    result = {
        'width': width[(data_type, instruction_set)],
        'intwidth': width[('int', instruction_set)],
        'bytes': 4 * width[("float", instruction_set)]
    }
    pre = prefix[instruction_set]
    for intrinsic_id, function_shortcut in base_names.items():
        function_shortcut = function_shortcut.strip()
        name = function_shortcut[:function_shortcut.index('[')]

        if 'Int' in intrinsic_id:
            suf = suffix['int']
            arg_string = get_argument_string(intrinsic_id, result['intwidth'], function_shortcut)
        else:
            suf = suffix[data_type]
            arg_string = get_argument_string(intrinsic_id, result['width'], function_shortcut)

        mask_suffix = '_mask' if instruction_set == 'avx512' and intrinsic_id in comparisons.keys() else ''
        result[intrinsic_id] = pre + "_" + name + "_" + suf + mask_suffix + arg_string

    bit_width = result['width'] * (64 if data_type == 'double' else 32)
    result['double'] = f"__m{bit_width}d"
    result['float'] = f"__m{bit_width}"
    result['int'] = f"__m{bit_width}i"
    result['bool'] = result[data_type]

    result['headers'] = headers[instruction_set]
    result['any'] = f"{pre}_movemask_{suf}({{0}}) > 0"
    result['all'] = f"{pre}_movemask_{suf}({{0}}) == {hex(2**result['width']-1)}"

    if instruction_set == 'avx512':
        size = result['width']
        result['&'] = f'_kand_mask{size}({{0}}, {{1}})'
        result['|'] = f'_kor_mask{size}({{0}}, {{1}})'
        result['any'] = f'!_ktestz_mask{size}_u8({{0}}, {{0}})'
        result['all'] = f'_kortestc_mask{size}_u8({{0}}, {{0}})'
        result['blendv'] = f'{pre}_mask_blend_{suf}({{2}}, {{0}}, {{1}})'
        result['rsqrt'] = f"{pre}_rsqrt14_{suf}({{0}})"
        result['abs'] = f"{pre}_abs_{suf}({{0}})"
        result['bool'] = f"__mmask{size}"

        params = " | ".join(["({{{i}}} ? {power} : 0)".format(i=i, power=2 ** i) for i in range(8)])
        result['makeVecBool'] = f"__mmask8(({params}) )"
        params = " | ".join(["({{0}} ? {power} : 0)".format(power=2 ** i) for i in range(8)])
        result['makeVecConstBool'] = f"__mmask8(({params}) )"

        vindex = f'{pre}_set_epi{bit_width//size}(' + ', '.join([str(i) for i in range(result['width'])][::-1]) + ')'
        vindex = f'{pre}_mullo_epi{bit_width//size}({vindex}, {pre}_set1_epi{bit_width//size}({{0}}))'
        result['storeS'] = f'{pre}_i{bit_width//size}scatter_{suf}({{0}}, ' + vindex.format("{2}") + \
                           f', {{1}}, {64//size})'
        result['maskStoreS'] = f'{pre}_mask_i{bit_width//size}scatter_{suf}({{0}}, {{3}}, ' + vindex.format("{2}") + \
                               f', {{1}}, {64//size})'
        result['loadS'] = f'{pre}_i{bit_width//size}gather_{suf}(' + vindex.format("{1}") + f', {{0}}, {64//size})'

    if instruction_set == 'avx' and data_type == 'float':
        result['rsqrt'] = f"{pre}_rsqrt_{suf}({{0}})"

    result['+int'] = f"{pre}_add_{suffix['int']}({{0}}, {{1}})"

    result['streamFence'] = '_mm_mfence()'

    return result
