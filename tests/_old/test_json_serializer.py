"""
Test the pystencils-specific JSON encoder and serializer as used in the Database class.
"""

import numpy as np
import tempfile

from pystencils.config import CreateKernelConfig
from pystencils import Target, Field
from pystencils.runhelper.db import Database, PystencilsJsonSerializer


def test_json_serializer():

    dtype = np.float32

    index_arr = np.zeros((3,), dtype=dtype)
    indexed_field = Field.create_from_numpy_array('index', index_arr)

    # create pystencils config
    config = CreateKernelConfig(target=Target.CPU, function_name='dummy_config', data_type=dtype,
                                index_fields=[indexed_field])

    # create dummy database
    temp_dir = tempfile.TemporaryDirectory()
    db = Database(file=temp_dir.name, serializer_info=('pystencils_serializer', PystencilsJsonSerializer))

    db.save(params={'config': config}, result={'test': 'dummy'})
