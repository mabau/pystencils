import subprocess
import os


def getEnvironment(versionSpecifier, arch='x64'):
    """
    Returns an environment dictionary, for activating the Visual Studio compiler
    :param versionSpecifier: either a version number, year number, 'auto' or 'latest' for automatic detection of latest
                             installed version or 'setuptools' for setuptools-based detection
    :param arch: x86 or x64
    """
    if versionSpecifier == 'setuptools':
        return getEnvironmentFromSetupTools(arch)
    elif '\\' in versionSpecifier:
        vcVarsPath = findVcVarsAllViaFilesystemSearch(versionSpecifier)
        return getEnvironmentFromVcVarsFile(vcVarsPath, arch)
    else:
        try:
            if versionSpecifier in ('auto', 'latest'):
                versionNr = findLatestMsvcVersionUsingEnvironmentVariables()
            else:
                versionNr = normalizeMsvcVersion(versionSpecifier)
            vcVarsPath = getVcVarsPathViaEnvironmentVariable(versionNr)
        except ValueError:
            vcVarsPath = findVcVarsAllViaFilesystemSearch("C:\\Program Files (x86)\\Microsoft Visual Studio")
            if vcVarsPath is None:
                vcVarsPath = findVcVarsAllViaFilesystemSearch("C:\\Program Files\\Microsoft Visual Studio")
            if vcVarsPath is None:
                raise ValueError("Visual Studio not found. Write path to VS folder in pystencils config")

        return getEnvironmentFromVcVarsFile(vcVarsPath, arch)


def findLatestMsvcVersionUsingEnvironmentVariables():
    import re
    regex = re.compile('VS(\d\d)\dCOMNTOOLS')
    versions = []
    for key, value in os.environ.items():
        match = regex.match(key)
        if match:
            versions.append(int(match.group(1)))
    if len(versions) == 0:
        raise ValueError("Visual Studio not found.")
    versions.sort()
    return versions[-1]


def normalizeMsvcVersion(version):
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


def getEnvironmentFromVcVarsFile(vcVarsFile, arch):
    out = subprocess.check_output(
        'cmd /u /c "{}" {} && set'.format(vcVarsFile, arch),
        stderr=subprocess.STDOUT,
    ).decode('utf-16le', errors='replace')

    env = {key.upper(): value for key, _, value in (line.partition('=') for line in out.splitlines()) if key and value}
    return env


def getVcVarsPathViaEnvironmentVariable(versionNr):
    environmentVarName = 'VS%d0COMNTOOLS' % (versionNr,)
    vcPath = os.environ[environmentVarName]
    path = os.path.join(vcPath, '..', '..', 'VC', 'vcvarsall.bat')
    return os.path.abspath(path)


def getEnvironmentFromSetupTools(arch):
    from setuptools.msvc import msvc14_get_vc_env
    msvcEnv = msvc14_get_vc_env(arch)
    return {k.upper(): v for k, v in msvcEnv.items()}


def findVcVarsAllViaFilesystemSearch(basePath):
    matches = []
    for root, dirnames, filenames in os.walk(basePath):
        for filename in filenames:
            if filename == 'vcvarsall.bat':
                matches.append(os.path.join(root, filename))

    matches.sort(reverse=True)
    if matches:
        return matches[0]
