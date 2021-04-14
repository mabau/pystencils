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


def get_vector_instruction_set_ppc(data_type='double', instruction_set='vsx'):
    if instruction_set != 'vsx':
        raise NotImplementedError(instruction_set)

    base_names = {
        '+': 'add[0, 1]',
        '-': 'sub[0, 1]',
        '*': 'mul[0, 1]',
        '/': 'div[0, 1]',
        'sqrt': 'sqrt[0]',
        'rsqrt': 'rsqrte[0]',  # rsqrt is available too, but not on Clang

        'loadU': 'xl[0x0, 0]',
        'loadA': 'ld[0x0, 0]',
        'storeU': 'xst[1, 0x0, 0]',
        'storeA': 'st[1, 0x0, 0]',
        'storeAAndFlushCacheline': 'stl[1, 0x0, 0]',

        'abs': 'abs[0]',
        '==': 'cmpeq[0, 1]',
        '!=': 'cmpne[0, 1]',
        '<=': 'cmple[0, 1]',
        '<': 'cmplt[0, 1]',
        '>=': 'cmpge[0, 1]',
        '>': 'cmpgt[0, 1]',
        '&': 'and[0, 1]',
        '|': 'or[0, 1]',
        'blendv': 'sel[0, 1, 2]',

        ('any', '=='): 'any_eq[0, 1]',
        ('any', '!='): 'any_ne[0, 1]',
        ('any', '<='): 'any_le[0, 1]',
        ('any', '<'): 'any_lt[0, 1]',
        ('any', '>='): 'any_ge[0, 1]',
        ('any', '>'): 'any_gt[0, 1]',
        ('all', '=='): 'all_eq[0, 1]',
        ('all', '!='): 'all_ne[0, 1]',
        ('all', '<='): 'all_le[0, 1]',
        ('all', '<'): 'all_lt[0, 1]',
        ('all', '>='): 'all_ge[0, 1]',
        ('all', '>'): 'all_gt[0, 1]',
    }

    bits = {'double': 64,
            'float': 32,
            'int': 32}

    width = 128 // bits[data_type]
    intwidth = 128 // bits['int']

    result = dict()
    result['bytes'] = 16

    for intrinsic_id, function_shortcut in base_names.items():
        function_shortcut = function_shortcut.strip()
        name = function_shortcut[:function_shortcut.index('[')]

        arg_string = get_argument_string(function_shortcut)

        result[intrinsic_id] = 'vec_' + name + arg_string

    if data_type == 'double':
        # Clang and XL C++ are missing these for doubles
        result['loadA'] = '(__vector double)' + result['loadA'].format('(float*) {0}')
        result['storeA'] = result['storeA'].format('(float*) {0}', '(__vector float) {1}')
        result['storeAAndFlushCacheline'] = result['storeAAndFlushCacheline'].format('(float*) {0}',
                                                                                     '(__vector float) {1}')

    result['+int'] = "vec_add({0}, {1})"

    result['width'] = width
    result['intwidth'] = intwidth
    result[data_type] = f'__vector {data_type}'
    result['int'] = '__vector int'
    result['bool'] = f'__vector __bool {"long long" if data_type == "double" else "int"}'
    result['headers'] = ['<altivec.h>', '"ppc_altivec_helpers.h"']

    result['makeVecConst'] = '((' + result[data_type] + '){{' + \
        ", ".join(['(' + data_type + ') {0}' for _ in range(width)]) + '}})'
    result['makeVec'] = '((' + result[data_type] + '){{' + \
        ", ".join(['{' + data_type + '} {' + str(i) + '}' for i in range(width)]) + '}})'
    result['makeVecConstInt'] = '((' + result['int'] + '){{' + ", ".join(['(int) {0}' for _ in range(intwidth)]) + '}})'
    result['makeVecInt'] = '((' + result['int'] + '){{(int) {0}, (int) {1}, (int) {2}, (int) {3}}})'

    result['any'] = 'vec_any_ne({0}, ((' + result['bool'] + ') {{' + ", ".join(['0'] * width) + '}}))'
    result['all'] = 'vec_all_ne({0}, ((' + result['bool'] + ') {{' + ", ".join(['0'] * width) + '}}))'

    result['cachelineSize'] = 'cachelineSize()'
    result['cachelineZero'] = 'cachelineZero((void*) {0})'

    return result
