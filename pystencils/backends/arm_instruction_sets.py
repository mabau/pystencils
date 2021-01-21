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


def get_vector_instruction_set_arm(data_type='double', instruction_set='neon', q_registers=True):
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
        # '&': 'and[0, 1]', -> only for integer values available
        # '|': 'orr[0, 1]'

    }

    bits = {'double': 64,
            'float': 32}

    if q_registers is True:
        q_reg = 'q'
        width = 128 // bits[data_type]
        suffix = f'q_f{bits[data_type]}'
    else:
        q_reg = ''
        width = 64 // bits[data_type]
        suffix = f'_f{bits[data_type]}'

    result = dict()

    for intrinsic_id, function_shortcut in base_names.items():
        function_shortcut = function_shortcut.strip()
        name = function_shortcut[:function_shortcut.index('[')]

        arg_string = get_argument_string(function_shortcut)

        result[intrinsic_id] = 'v' + name + suffix + arg_string

    result['makeVecConst'] = 'vdup' + q_reg + '_n_f' + str(bits[data_type]) + '({0})'
    result['makeVec'] = 'vdup' + q_reg + '_n_f' + str(bits[data_type]) + '({0})'

    result['rsqrt'] = None

    result['width'] = width
    result['double'] = 'float64x' + str(width) + '_t'
    result['float'] = 'float32x' + str(width * 2) + '_t'
    result['headers'] = ['<arm_neon.h>']

    result['!='] = 'vmvnq_u%d(%s)' % (bits[data_type], result['=='])

    return result
