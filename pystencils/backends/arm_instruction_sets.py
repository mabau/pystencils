def get_argument_string(function_shortcut):
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


def get_vector_instruction_set_arm(data_type='double', instruction_set='neon'):
    if instruction_set != 'neon':
        raise NotImplementedError(instruction_set)

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
        'stream': 'st1[0, 1]',

        'abs': 'abs[0]',
        '==': 'ceq[0, 1]',
        '<=': 'cle[0, 1]',
        '<': 'clt[0, 1]',
        '>=': 'cge[0, 1]',
        '>': 'cgt[0, 1]',
    }

    bits = {'double': 64,
            'float': 32,
            'int': 32}

    width = 128 // bits[data_type]
    intwidth = 128 // bits['int']
    suffix = f'q_f{bits[data_type]}'

    result = dict()

    for intrinsic_id, function_shortcut in base_names.items():
        function_shortcut = function_shortcut.strip()
        name = function_shortcut[:function_shortcut.index('[')]

        arg_string = get_argument_string(function_shortcut)

        result[intrinsic_id] = 'v' + name + suffix + arg_string

    result['makeVecConst'] = f'vdupq_n_f{bits[data_type]}' + '({0})'
    result['makeVec'] = f'makeVec_f{bits[data_type]}' + '(' + ", ".join(['{' + str(i) + '}' for i in range(width)]) + \
        ')'
    result['makeVecConstInt'] = f'vdupq_n_s{bits["int"]}' + '({0})'
    result['makeVecInt'] = f'makeVec_s{bits["int"]}' + '({0}, {1}, {2}, {3})'

    result['+int'] = f"vaddq_s{bits['int']}" + "({0}, {1})"

    result['rsqrt'] = None

    result['width'] = width
    result['intwidth'] = intwidth
    result[data_type] = f'float{bits[data_type]}x{width}_t'
    result['int'] = f'int{bits["int"]}x{bits[data_type]}_t'
    result['bool'] = f'uint{bits[data_type]}x{width}_t'
    result['headers'] = ['<arm_neon.h>', '"arm_neon_helpers.h"']

    result['!='] = f'vmvnq_u{bits[data_type]}({result["=="]})'

    result['&'] = f'vandq_u{bits[data_type]}' + '({0}, {1})'
    result['|'] = f'vorrq_u{bits[data_type]}' + '({0}, {1})'
    result['blendv'] = f'vbslq_f{bits[data_type]}' + '({2}, {1}, {0})'
    result['any'] = f'vaddlvq_u8(vreinterpretq_u8_u{bits[data_type]}({{0}})) > 0'
    result['all'] = f'vaddlvq_u8(vreinterpretq_u8_u{bits[data_type]}({{0}})) == 16*0xff'

    return result
