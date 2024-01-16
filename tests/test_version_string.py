import pystencils as ps


def test_version_string():
    version = ps.__version__
    print(version)

    numeric_version = [int(x, 10) for x in version.split('.')[0:1]]
    test_version = sum(x * (100 ** i) for i, x in enumerate(numeric_version))

    assert test_version >= 1
