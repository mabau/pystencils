from os.path import dirname, join, realpath


def get_pystencils_include_path():
    return dirname(realpath(__file__))


def get_pycuda_include_path():
    import pycuda
    return join(dirname(realpath(pycuda.__file__)), 'cuda')
