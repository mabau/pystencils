from os.path import dirname, realpath


def get_pystencils_include_path():
    return dirname(realpath(__file__))
