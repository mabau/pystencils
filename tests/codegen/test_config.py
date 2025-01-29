import pytest

from dataclasses import dataclass
import numpy as np
from pystencils.codegen.config import (
    BasicOption,
    Option,
    Category,
    ConfigBase,
    CreateKernelConfig,
    CpuOptions
)
from pystencils.field import Field, FieldType
from pystencils.types.quick import Int, UInt, Fp, Ptr
from pystencils.types import PsVectorType


def test_descriptors():

    @dataclass
    class SampleCategory(ConfigBase):
        val1: BasicOption[int] = BasicOption(2)
        val2: Option[bool, str | bool] = Option(False)

        @val2.validate
        def validate_val2(self, v: str | bool):
            if isinstance(v, str):
                if v.lower() in ("off", "false", "no"):
                    return False
                elif v.lower() in ("on", "true", "yes"):
                    return True

                raise ValueError()
            else:
                return v

    @dataclass
    class SampleConfig(ConfigBase):
        cat: Category[SampleCategory] = Category(SampleCategory())
        val: BasicOption[str] = BasicOption("fallback")

    cfg = SampleConfig()

    #   Check unset and default values
    assert cfg.val is None
    assert cfg.get_option("val") == "fallback"

    #   Check setting
    cfg.val = "test"
    assert cfg.val == "test"
    assert cfg.get_option("val") == "test"
    assert cfg.is_option_set("val")

    #   Check unsetting
    cfg.val = None
    assert not cfg.is_option_set("val")
    assert cfg.val is None

    #   Check category
    assert cfg.cat.val1 is None
    assert cfg.cat.get_option("val1") == 2
    assert cfg.cat.val2 is None
    assert cfg.cat.get_option("val2") is False

    #   Check copy on category setting
    c = SampleCategory(32, "on")
    cfg.cat = c
    assert cfg.cat.val1 == 32
    assert cfg.cat.val2 is True

    assert cfg.cat is not c
    c.val1 = 13
    assert cfg.cat.val1 == 32

    #   Check that category objects on two config objects are not the same
    cfg1 = SampleConfig()
    cfg2 = SampleConfig()

    assert cfg1.cat is not cfg2.cat


def test_category_init():
    cfg1 = CreateKernelConfig()
    cfg2 = CreateKernelConfig()

    assert cfg1.cpu is not cfg2.cpu
    assert cfg1.cpu.openmp is not cfg2.cpu.openmp
    assert cfg1.cpu.vectorize is not cfg2.cpu.vectorize
    assert cfg1.gpu is not cfg2.gpu


def test_category_copy():
    cfg = CreateKernelConfig()
    cpu_repl = CpuOptions()
    cpu_repl.openmp.num_threads = 42

    cfg.cpu = cpu_repl
    assert cfg.cpu.openmp.num_threads == 42
    assert cfg.cpu is not cpu_repl
    assert cfg.cpu.openmp is not cpu_repl.openmp


def test_config_validation():
    #   Check index dtype validation
    cfg = CreateKernelConfig(index_dtype="int32")
    assert cfg.index_dtype == Int(32)
    cfg.index_dtype = np.uint64
    assert cfg.index_dtype == UInt(64)

    with pytest.raises(ValueError):
        _ = CreateKernelConfig(index_dtype=np.float32)

    with pytest.raises(ValueError):
        cfg.index_dtype = "double"

    #   Check default dtype validation
    cfg = CreateKernelConfig(default_dtype="float32")
    assert cfg.default_dtype == Fp(32)
    cfg.default_dtype = np.int64
    assert cfg.default_dtype == Int(64)

    with pytest.raises(ValueError):
        cfg.default_dtype = PsVectorType(Fp(64), 4)

    with pytest.raises(ValueError):
        _ = CreateKernelConfig(default_dtype=Ptr(Fp(32)))

    #   Check index field validation
    idx_field = Field.create_generic(
        "idx", spatial_dimensions=1, field_type=FieldType.INDEXED
    )
    cfg.index_field = idx_field
    assert cfg.index_field == idx_field

    with pytest.raises(ValueError):
        cfg.index_field = Field.create_generic(
            "idx", spatial_dimensions=1, field_type=FieldType.GENERIC
        )


def test_override():
    cfg1 = CreateKernelConfig()
    cfg1.function_name = "test"
    cfg1.cpu.openmp.schedule = "dynamic"
    cfg1.gpu.manual_launch_grid = False
    cfg1.allow_double_writes = True

    cfg2 = CreateKernelConfig()
    cfg2.function_name = "func"
    cfg2.cpu.openmp.schedule = "static(5)"
    cfg2.cpu.vectorize.lanes = 12
    cfg2.allow_double_writes = False

    cfg1.override(cfg2)

    assert cfg1.function_name == "func"
    assert cfg1.cpu.openmp.schedule == "static(5)"
    assert cfg1.cpu.openmp.enable is None
    assert cfg1.cpu.vectorize.lanes == 12
    assert cfg1.cpu.vectorize.assume_aligned is None
    assert cfg1.allow_double_writes is False
