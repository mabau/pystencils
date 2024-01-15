import os

from setuptools import Extension, setup

import versioneer

def get_cmdclass():
    return versioneer.get_cmdclass()

setup(
    version=versioneer.get_version(),
    cmdclass=get_cmdclass(),
)
