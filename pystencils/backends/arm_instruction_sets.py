def get_argument_string(function_shortcut, first=''):
    args = function_shortcut[function_shortcut.index('[') + 1: -1]
    arg_string = "("
    if first:
        arg_string += first + ', '
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


def get_vector_instruction_set_arm(data_type='double', instruction_set='neon'):
    if instruction_set != 'neon' and not instruction_set.startswith('sve'):
        raise NotImplementedError(instruction_set)
    if instruction_set == 'sve':
        raise NotImplementedError("sizeless SVE is not implemented")

    if instruction_set.startswith('sve'):
        cmp = 'cmp'
        bitwidth = int(instruction_set[3:])
    elif instruction_set == 'neon':
        cmp = 'c'
        bitwidth = 128

    base_names = {
        '+': 'add[0, 1]',
        '-': 'sub[0, 1]',
        '*': 'mul[0, 1]',
        '/': 'div[0, 1]',
        'sqrt': 'sqrt[0]',

        'loadU': 'ld1[0]',
        'loadA': 'ld1[0]',
        'storeU': 'st1[0, 1]',
        'storeA': 'st1[0, 1]',

        'abs': 'abs[0]',
        '==': f'{cmp}eq[0, 1]',
        '!=': f'{cmp}eq[0, 1]',
        '<=': f'{cmp}le[0, 1]',
        '<': f'{cmp}lt[0, 1]',
        '>=': f'{cmp}ge[0, 1]',
        '>': f'{cmp}gt[0, 1]',
    }

    bits = {'double': 64,
            'float': 32,
            'int': 32}

    width = bitwidth // bits[data_type]
    intwidth = bitwidth // bits['int']
    if instruction_set.startswith('sve'):
        prefix = 'sv'
        suffix = f'_f{bits[data_type]}' 
    elif instruction_set == 'neon':
        prefix = 'v'
        suffix = f'q_f{bits[data_type]}' 

    result = dict()
    result['bytes'] = bitwidth // 8

    predicate = f'{prefix}whilelt_b{bits[data_type]}(0, {width})'
    int_predicate = f'{prefix}whilelt_b{bits["int"]}(0, {intwidth})'

    for intrinsic_id, function_shortcut in base_names.items():
        function_shortcut = function_shortcut.strip()
        name = function_shortcut[:function_shortcut.index('[')]

        arg_string = get_argument_string(function_shortcut, first=predicate if prefix == 'sv' else '')
        if prefix == 'sv' and not name.startswith('ld') and not name.startswith('st') and not name.startswith(cmp):
            undef = '_x'
        else:
            undef = ''

        result[intrinsic_id] = prefix + name + suffix + undef + arg_string

    result['width'] = width
    result['intwidth'] = intwidth

    if instruction_set.startswith('sve'):
        result['makeVecConst'] = f'svdup_f{bits[data_type]}' + '({0})'
        result['makeVecConstInt'] = f'svdup_s{bits["int"]}' + '({0})'
        result['makeVecIndex'] = f'svindex_s{bits["int"]}' + '({0}, {1})'

        vindex = f'svindex_u{bits[data_type]}(0, {{0}})'
        result['scatter'] = f'svst1_scatter_u{bits[data_type]}index_f{bits[data_type]}({predicate}, {{0}}, ' + \
                            vindex.format("{2}") + ', {1})'
        result['gather'] = f'svld1_gather_u{bits[data_type]}index_f{bits[data_type]}({predicate}, {{0}}, ' + \
                           vindex.format("{1}") + ')'

        result['+int'] = f"svadd_s{bits['int']}_x({int_predicate}, " + "{0}, {1})"

        result['float'] = 'svfloat32_st'
        result['double'] = 'svfloat64_st'
        result['int'] = f'svint{bits["int"]}_st'
        result['bool'] = 'svbool_st'

        result['headers'] = ['<arm_sve.h>', '"arm_neon_helpers.h"']

        result['&'] = f'svand_b_z({predicate},' + ' {0}, {1})'
        result['|'] = f'svorr_b_z({predicate},' + ' {0}, {1})'
        result['blendv'] = f'svsel_f{bits[data_type]}' + '({2}, {1}, {0})'
        result['any'] = f'svptest_any({predicate}, {{0}})'
        result['all'] = f'svcntp_b{bits[data_type]}({predicate}, {{0}}) == {width}'

        result['maskStoreU'] = result['storeU'].replace(predicate, '{2}')
        result['maskStoreA'] = result['storeA'].replace(predicate, '{2}')
        result['maskScatter'] = result['scatter'].replace(predicate, '{3}')

        result['compile_flags'] = [f'-msve-vector-bits={bitwidth}']
    else:
        result['makeVecConst'] = f'vdupq_n_f{bits[data_type]}' + '({0})'
        result['makeVec'] = f'makeVec_f{bits[data_type]}' + '(' + ", ".join(['{' + str(i) + '}' for i in
                                                                             range(width)]) + ')'
        result['makeVecConstInt'] = f'vdupq_n_s{bits["int"]}' + '({0})'
        result['makeVecInt'] = f'makeVec_s{bits["int"]}' + '({0}, {1}, {2}, {3})'

        result['+int'] = f"vaddq_s{bits['int']}" + "({0}, {1})"

        result[data_type] = f'float{bits[data_type]}x{width}_t'
        result['int'] = f'int{bits["int"]}x{intwidth}_t'
        result['bool'] = f'uint{bits[data_type]}x{width}_t'

        result['headers'] = ['<arm_neon.h>', '"arm_neon_helpers.h"']

        result['!='] = f'vmvnq_u{bits[data_type]}({result["=="]})'

        result['&'] = f'vandq_u{bits[data_type]}' + '({0}, {1})'
        result['|'] = f'vorrq_u{bits[data_type]}' + '({0}, {1})'
        result['blendv'] = f'vbslq_f{bits[data_type]}' + '({2}, {1}, {0})'
        result['any'] = f'vaddlvq_u8(vreinterpretq_u8_u{bits[data_type]}({{0}})) > 0'
        result['all'] = f'vaddlvq_u8(vreinterpretq_u8_u{bits[data_type]}({{0}})) == 16*0xff'

    if bitwidth & (bitwidth - 1) == 0:
        # only power-of-2 vector sizes will evenly divide a cacheline
        result['cachelineSize'] = 'cachelineSize()'
        result['cachelineZero'] = 'cachelineZero((void*) {0})'

    return result
