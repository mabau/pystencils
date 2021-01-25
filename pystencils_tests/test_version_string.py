import pystencils as ps
from pathlib import Path

def test_version_string():
    file_path = Path(__file__).parent
    release_version = file_path.parent.absolute() / 'RELEASE-VERSION'
    if release_version.exists ():
        with open(release_version, "r") as f:
            version = f.read()
        assert ps.__version__ == version
    else:
        assert ps.__version__ == "development"
