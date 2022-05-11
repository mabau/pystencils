import os
import runpy
import sys
import tempfile
import warnings

import nbformat
import pytest
from nbconvert import PythonExporter

from pystencils.boundaries.createindexlistcython import *  # NOQA
# Trigger config file reading / creation once - to avoid race conditions when multiple instances are creating it
# at the same time
from pystencils.cpu import cpujit

# trigger cython imports - there seems to be a problem when multiple processes try to compile the same cython file
# at the same time
try:
    import pyximport
    pyximport.install(language_level=3)
except ImportError:
    pass


SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath('pystencils'))

# the Ubuntu pipeline uses an older version of pytest which uses deprecated functionality.
# This leads to many warinings in the test and coverage pipeline.
pytest_numeric_version = [int(x, 10) for x in pytest.__version__.split('.')]
pytest_numeric_version.reverse()
pytest_version = sum(x * (100 ** i) for i, x in enumerate(pytest_numeric_version))


def add_path_to_ignore(path):
    if not os.path.exists(path):
        return
    global collect_ignore
    collect_ignore += [os.path.join(SCRIPT_FOLDER, path, f) for f in os.listdir(os.path.join(SCRIPT_FOLDER, path))]


collect_ignore = [os.path.join(SCRIPT_FOLDER, "doc", "conf.py"),
                  os.path.join(SCRIPT_FOLDER, "pystencils", "opencl", "opencl.autoinit")]
add_path_to_ignore('pystencils_tests/benchmark')
add_path_to_ignore('_local_tmp')


try:
    import pycuda
except ImportError:
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "pystencils_tests/test_cudagpu.py")]
    add_path_to_ignore('pystencils/gpucuda')

try:
    import waLBerla
except ImportError:
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "pystencils_tests/test_aligned_array.py"),
                       os.path.join(SCRIPT_FOLDER, "pystencils_tests/test_datahandling_parallel.py"),
                       os.path.join(SCRIPT_FOLDER, "doc/notebooks/03_tutorial_datahandling.ipynb"),
                       os.path.join(SCRIPT_FOLDER, "pystencils/datahandling/parallel_datahandling.py"),
                       os.path.join(SCRIPT_FOLDER, "pystencils_tests/test_small_block_benchmark.ipynb")]

try:
    import blitzdb
except ImportError:
    add_path_to_ignore('pystencils/runhelper')
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "pystencils_tests/test_parameterstudy.py")]

try:
    import islpy
except ImportError:
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "pystencils/integer_set_analysis.py")]

try:
    import graphviz
except ImportError:
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "pystencils/backends/dot.py")]
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "doc/notebooks/01_tutorial_getting_started.ipynb")]

try:
    import pyevtk
except ImportError:
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "pystencils/datahandling/vtk.py")]

collect_ignore += [os.path.join(SCRIPT_FOLDER, 'setup.py')]

for root, sub_dirs, files in os.walk('.'):
    for f in files:
        if f.endswith(".ipynb") and not any(f.startswith(k) for k in ['demo', 'tutorial', 'test', 'doc']):
            collect_ignore.append(f)


class IPythonMockup:
    def run_line_magic(self, *args, **kwargs):
        pass

    def run_cell_magic(self, *args, **kwargs):
        pass

    def magic(self, *args, **kwargs):
        pass

    def __bool__(self):
        return False


class IPyNbTest(pytest.Item):
    def __init__(self, name, parent, code):
        super(IPyNbTest, self).__init__(name, parent)
        self.code = code
        self.add_marker('notebook')

    @pytest.mark.filterwarnings("ignore:IPython.core.inputsplitter is deprecated")
    def runtest(self):
        global_dict = {'get_ipython': lambda: IPythonMockup(),
                       'is_test_run': True}

        # disable matplotlib output
        exec("import matplotlib.pyplot as p; "
             "p.switch_backend('Template')", global_dict)

        # in notebooks there is an implicit plt.show() - if this is not called a warning is shown when the next
        # plot is created. This warning is suppressed here
        exec("import warnings;"
             "warnings.filterwarnings('ignore', 'Adding an axes using the same arguments as a previous.*')",
             global_dict)
        with tempfile.NamedTemporaryFile() as f:
            f.write(self.code.encode())
            f.flush()
            runpy.run_path(f.name, init_globals=global_dict, run_name=self.name)


class IPyNbFile(pytest.File):
    def collect(self):
        exporter = PythonExporter()
        exporter.exclude_markdown = True
        exporter.exclude_input_prompt = True

        notebook_contents = self.fspath.open(encoding='utf-8')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "IPython.core.inputsplitter is deprecated")
            notebook = nbformat.read(notebook_contents, 4)
            code, _ = exporter.from_notebook_node(notebook)
        if pytest_version >= 50403:
            yield IPyNbTest.from_parent(name=self.name, parent=self, code=code)
        else:
            yield IPyNbTest(self.name, self, code)

    def teardown(self):
        pass


def pytest_collect_file(path, parent):
    glob_exprs = ["*demo*.ipynb", "*tutorial*.ipynb", "test_*.ipynb"]
    if any(path.fnmatch(g) for g in glob_exprs):
        if pytest_version >= 50403:
            return IPyNbFile.from_parent(fspath=path, parent=parent)
        else:
            return IPyNbFile(path, parent)
