import os
import subprocess


def get_environment(version_specifier, arch='x64'):
    """Returns an environment dictionary, for activating the Visual Studio compiler.

    Args:
        version_specifier: either a version number, year number, 'auto' or 'latest' for automatic detection of latest
                          installed version or 'setuptools' for setuptools-based detection
        arch: x86 or x64
    """
    if version_specifier == 'setuptools':
        return get_environment_from_setup_tools(arch)
    elif '\\' in version_specifier:
        vc_vars_path = find_vc_vars_all_via_filesystem_search(version_specifier)
        return get_environment_from_vc_vars_file(vc_vars_path, arch)
    else:
        try:
            if version_specifier in ('auto', 'latest'):
                version_nr = find_latest_msvc_version_using_environment_variables()
            else:
                version_nr = normalize_msvc_version(version_specifier)
            vc_vars_path = get_vc_vars_path_via_environment_variable(version_nr)
        except ValueError:
            vc_vars_path = find_vc_vars_all_via_filesystem_search("C:\\Program Files (x86)\\Microsoft Visual Studio")
            if vc_vars_path is None:
                vc_vars_path = find_vc_vars_all_via_filesystem_search("C:\\Program Files\\Microsoft Visual Studio")
            if vc_vars_path is None:
                raise ValueError("Visual Studio not found. Write path to VS folder in pystencils config")

        return get_environment_from_vc_vars_file(vc_vars_path, arch)


def find_latest_msvc_version_using_environment_variables():
    import re
    # noinspection SpellCheckingInspection
    regex = re.compile(r'VS(\d\d)\dCOMNTOOLS')
    versions = []
    for key, value in os.environ.items():
        match = regex.match(key)
        if match:
            versions.append(int(match.group(1)))
    if len(versions) == 0:
        raise ValueError("Visual Studio not found.")
    versions.sort()
    return versions[-1]


def normalize_msvc_version(version):
    """
    Takes version specifiers in the following form:
        - year: 2012, 2013, 2015, either as int or string
        - version numbers with or without dot i.e. 11.0 or 11
    :return: integer version number
    """
    if isinstance(version, str) and '.' in version:
        version = version.split('.')[0]

    version = int(version)
    mapping = {
        2015: 14,
        2013: 12,
        2012: 11
    }
    if version in mapping:
        return mapping[version]
    else:
        return version


def get_environment_from_vc_vars_file(vc_vars_file, arch):
    out = subprocess.check_output(
        f'cmd /u /c "{vc_vars_file}" {arch} && set',
        stderr=subprocess.STDOUT,
    ).decode('utf-16le', errors='replace')

    env = {key.upper(): value for key, _, value in (line.partition('=') for line in out.splitlines()) if key and value}
    return env


def get_vc_vars_path_via_environment_variable(version_nr):
    # noinspection SpellCheckingInspection
    environment_var_name = 'VS%d0COMNTOOLS' % (version_nr,)
    vc_path = os.environ[environment_var_name]
    path = os.path.join(vc_path, '..', '..', 'VC', 'vcvarsall.bat')
    return os.path.abspath(path)


def get_environment_from_setup_tools(arch):
    from setuptools.msvc import msvc14_get_vc_env
    msvc_env = msvc14_get_vc_env(arch)
    return {k.upper(): v for k, v in msvc_env.items()}


def find_vc_vars_all_via_filesystem_search(base_path):
    matches = []
    for root, dir_names, file_names in os.walk(base_path):
        for filename in file_names:
            if filename == 'vcvarsall.bat':
                matches.append(os.path.join(root, filename))

    matches.sort(reverse=True)
    if matches:
        return matches[0]
