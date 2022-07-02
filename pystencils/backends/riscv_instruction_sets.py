def get_argument_string(function_shortcut, last=''):
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
    if last:
        arg_string += last + ','
    arg_string = arg_string[:-1] + ")"
    return arg_string


def get_vector_instruction_set_riscv(data_type='double', instruction_set='rvv'):
    assert instruction_set == 'rvv'

    bits = {'double': 64,
            'float': 32,
            'int': 32}

    base_names = {
        '+': 'fadd_vv[0, 1]',
        '-': 'fsub_vv[0, 1]',
        '*': 'fmul_vv[0, 1]',
        '/': 'fdiv_vv[0, 1]',
        'sqrt': 'fsqrt_v[0]',

        'loadU': f'le{bits[data_type]}_v[0]',
        'loadA': f'le{bits[data_type]}_v[0]',
        'storeU': f'se{bits[data_type]}_v[0, 1]',
        'storeA': f'se{bits[data_type]}_v[0, 1]',
        'maskStoreU': f'se{bits[data_type]}_v[2, 0, 1]',
        'maskStoreA': f'se{bits[data_type]}_v[2, 0, 1]',
        'loadS': f'lse{bits[data_type]}_v[0, 1]',
        'storeS': f'sse{bits[data_type]}_v[0, 2, 1]',
        'maskStoreS': f'sse{bits[data_type]}_v[2, 0, 3, 1]',

        'abs': 'fabs_v[0]',
        '==': 'mfeq_vv[0, 1]',
        '!=': 'mfne_vv[0, 1]',
        '<=': 'mfle_vv[0, 1]',
        '<': 'mflt_vv[0, 1]',
        '>=': 'mfge_vv[0, 1]',
        '>': 'mfgt_vv[0, 1]',
        '&': 'mand_mm[0, 1]',
        '|': 'mor_mm[0, 1]',

        'blendv': 'merge_vvm[2, 0, 1]',
        'any': 'cpop_m[0]',
        'all': 'cpop_m[0]',
    }

    result = dict()

    width = f'vsetvlmax_e{bits[data_type]}m1()'
    intwidth = 'vsetvlmax_e{bits["int"]}m1()'
    result['bytes'] = 'vsetvlmax_e8m1()'
    prefix = 'v'
    suffix = f'_f{bits[data_type]}m1'

    vl = '{loop_stop} - {loop_counter}'
    int_vl = f'({vl})*{bits[data_type]//bits["int"]}'

    for intrinsic_id, function_shortcut in base_names.items():
        function_shortcut = function_shortcut.strip()
        name = function_shortcut[:function_shortcut.index('[')]
        if name.startswith('mf'):
            suffix2 = suffix + f'_b{bits[data_type]}'
        elif name.endswith('_mm') or name.endswith('_m'):
            suffix2 = f'_b{bits[data_type]}'
        elif intrinsic_id.startswith('mask'):
            suffix2 = suffix + '_m'
        else:
            suffix2 = suffix

        arg_string = get_argument_string(function_shortcut, last=vl)

        result[intrinsic_id] = prefix + name + suffix2 + arg_string

    from pystencils.backends.cbackend import CFunction
    result['width'] = CFunction(width, "int")
    result['intwidth'] = CFunction(intwidth, "int")

    result['makeVecConst'] = f'vfmv_v_f_f{bits[data_type]}m1({{0}}, {vl})'
    result['makeVecConstInt'] = f'vmv_v_x_i{bits["int"]}m1({{0}}, {int_vl})'
    result['makeVecIndex'] = f'vmacc_vx_i{bits["int"]}m1({result["makeVecConstInt"]}, {{1}}, ' + \
                             f'vid_v_i{bits["int"]}m1({int_vl}), {int_vl})'

    result['storeS'] = result['storeS'].replace('{2}', f'{{2}}*{bits[data_type]//8}')
    result['loadS'] = result['loadS'].replace('{1}', f'{{1}}*{bits[data_type]//8}')
    result['maskStoreS'] = result['maskStoreS'].replace('{3}', f'{{3}}*{bits[data_type]//8}')

    result['+int'] = f"vadd_vv_i{bits['int']}m1({{0}}, {{1}}, {int_vl})"

    result['float'] = f'vfloat{bits["float"]}m1_t'
    result['double'] = f'vfloat{bits["double"]}m1_t'
    result['int'] = f'vint{bits["int"]}m1_t'
    result['bool'] = f'vbool{bits[data_type]}_t'

    result['headers'] = ['<riscv_vector.h>']

    result['any'] += ' > 0x0'
    result['all'] += f' == vsetvl_e{bits[data_type]}m1({vl})'

    return result
