from setuptools import setup, __version__ as setuptools_version

if int(setuptools_version.split('.')[0]) < 61:
    raise Exception(
        "[ERROR] pystencils requires at least setuptools version 61 to install.\n"
        "If this error occurs during an installation via pip, it is likely that there is a conflict between "
        "versions of setuptools installed by pip and the system package manager. "
        "In this case, it is recommended to install pystencils into a virtual environment instead."
    )

import versioneer


def get_cmdclass():
    return versioneer.get_cmdclass()


setup(
    version=versioneer.get_version(),
    cmdclass=get_cmdclass(),
)
