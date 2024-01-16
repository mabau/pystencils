from collections import defaultdict
import numpy as np
import pytest

from pystencils import CreateKernelConfig, Target, Backend
from pystencils.typing import BasicType


def test_config():
    # targets
    config = CreateKernelConfig(target=Target.CPU)
    assert config.target == Target.CPU
    assert config.backend == Backend.C

    config = CreateKernelConfig(target=Target.GPU)
    assert config.target == Target.GPU
    assert config.backend == Backend.CUDA

    # typing
    config = CreateKernelConfig(data_type=np.float64)
    assert isinstance(config.data_type, defaultdict)
    assert config.data_type.default_factory() == BasicType('float64')
    assert config.default_number_float == BasicType('float64')
    assert config.default_number_int == BasicType('int64')

    config = CreateKernelConfig(data_type=np.float32)
    assert isinstance(config.data_type, defaultdict)
    assert config.data_type.default_factory() == BasicType('float32')
    assert config.default_number_float == BasicType('float32')
    assert config.default_number_int == BasicType('int64')

    config = CreateKernelConfig(data_type=np.float32, default_number_float=np.float64)
    assert isinstance(config.data_type, defaultdict)
    assert config.data_type.default_factory() == BasicType('float32')
    assert config.default_number_float == BasicType('float64')
    assert config.default_number_int == BasicType('int64')

    config = CreateKernelConfig(data_type=np.float32, default_number_float=np.float64, default_number_int=np.int16)
    assert isinstance(config.data_type, defaultdict)
    assert config.data_type.default_factory() == BasicType('float32')
    assert config.default_number_float == BasicType('float64')
    assert config.default_number_int == BasicType('int16')

    config = CreateKernelConfig(data_type='float64')
    assert isinstance(config.data_type, defaultdict)
    assert config.data_type.default_factory() == BasicType('float64')
    assert config.default_number_float == BasicType('float64')
    assert config.default_number_int == BasicType('int64')

    config = CreateKernelConfig(data_type={'a': np.float64, 'b': np.float32})
    assert isinstance(config.data_type, defaultdict)
    assert config.data_type.default_factory() == BasicType('float64')
    assert config.default_number_float == BasicType('float64')
    assert config.default_number_int == BasicType('int64')

    config = CreateKernelConfig(data_type={'a': np.float32, 'b': np.int32})
    assert isinstance(config.data_type, defaultdict)
    assert config.data_type.default_factory() == BasicType('float32')
    assert config.default_number_float == BasicType('float32')
    assert config.default_number_int == BasicType('int64')


def test_config_target_as_string():
    with pytest.raises(ValueError):
        CreateKernelConfig(target='cpu')


def test_config_backend_as_string():
    with pytest.raises(ValueError):
        CreateKernelConfig(backend='C')


def test_config_python_types():
    with pytest.raises(ValueError):
        CreateKernelConfig(data_type=float)


def test_config_python_types2():
    with pytest.raises(ValueError):
        CreateKernelConfig(data_type={'a': float})


def test_config_python_types3():
    with pytest.raises(ValueError):
        CreateKernelConfig(default_number_float=float)


def test_config_python_types4():
    with pytest.raises(ValueError):
        CreateKernelConfig(default_number_int=int)


def test_config_python_types5():
    with pytest.raises(ValueError):
        CreateKernelConfig(data_type="float")


def test_config_python_types6():
    with pytest.raises(ValueError):
        CreateKernelConfig(default_number_float="float")


def test_config_python_types7():
    dtype = defaultdict(lambda: 'float', {'a': np.float64, 'b': np.int64})
    with pytest.raises(ValueError):
        CreateKernelConfig(data_type=dtype)


def test_config_python_types8():
    dtype = defaultdict(lambda: float, {'a': np.float64, 'b': np.int64})
    with pytest.raises(ValueError):
        CreateKernelConfig(data_type=dtype)


def test_config_python_types9():
    dtype = defaultdict(lambda: 'float32', {'a': 'float', 'b': np.int64})
    with pytest.raises(ValueError):
        CreateKernelConfig(data_type=dtype)


def test_config_python_types10():
    dtype = defaultdict(lambda: 'float32', {'a': float, 'b': np.int64})
    with pytest.raises(ValueError):
        CreateKernelConfig(data_type=dtype)
