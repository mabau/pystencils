import pytest

from pystencils import Assignment, CreateKernelConfig, Target, fields, create_kernel, get_code_str


@pytest.mark.parametrize('target', (Target.CPU, Target.GPU))
def test_intermediate_base_pointer(target):
    x = fields(f'x: double[3d]')
    y = fields(f'y: double[3d]')
    update = Assignment(x.center, y.center)

    config = CreateKernelConfig(base_pointer_specification=[], target=target)
    ast = create_kernel(update, config=config)
    code = get_code_str(ast)

    # no intermediate base pointers are created
    assert "_data_x[_stride_x_0*ctr_0 + _stride_x_1*ctr_1 + _stride_x_2*ctr_2] = " \
           "_data_y[_stride_y_0*ctr_0 + _stride_y_1*ctr_1 + _stride_y_2*ctr_2];" in code


    config = CreateKernelConfig(base_pointer_specification=[[0]], target=target)
    ast = create_kernel(update, config=config)
    code = get_code_str(ast)

    # intermediate base pointers for y and z
    assert "double * RESTRICT  _data_x_10_20 = _data_x + _stride_x_1*ctr_1 + _stride_x_2*ctr_2;" in code
    assert " double * RESTRICT _data_y_10_20 = _data_y + _stride_y_1*ctr_1 + _stride_y_2*ctr_2;" in code
    assert "_data_x_10_20[_stride_x_0*ctr_0] = _data_y_10_20[_stride_y_0*ctr_0];" in code

    config = CreateKernelConfig(base_pointer_specification=[[1]], target=target)
    ast = create_kernel(update, config=config)
    code = get_code_str(ast)

    # intermediate base pointers for x and z
    assert "double * RESTRICT  _data_x_00_20 = _data_x + _stride_x_0*ctr_0 + _stride_x_2*ctr_2;" in code
    assert "double * RESTRICT _data_y_00_20 = _data_y + _stride_y_0*ctr_0 + _stride_y_2*ctr_2;" in code
    assert "_data_x_00_20[_stride_x_1*ctr_1] = _data_y_00_20[_stride_y_1*ctr_1];" in code

    config = CreateKernelConfig(base_pointer_specification=[[2]], target=target)
    ast = create_kernel(update, config=config)
    code = get_code_str(ast)

    # intermediate base pointers for x and y
    assert "double * RESTRICT  _data_x_00_10 = _data_x + _stride_x_0*ctr_0 + _stride_x_1*ctr_1;" in code
    assert "double * RESTRICT _data_y_00_10 = _data_y + _stride_y_0*ctr_0 + _stride_y_1*ctr_1;" in code
    assert "_data_x_00_10[_stride_x_2*ctr_2] = _data_y_00_10[_stride_y_2*ctr_2];" in code

    config = CreateKernelConfig(target=target)
    ast = create_kernel(update, config=config)
    code = get_code_str(ast)

    # by default no intermediate base pointers are created
    assert "_data_x[_stride_x_0*ctr_0 + _stride_x_1*ctr_1 + _stride_x_2*ctr_2] = " \
           "_data_y[_stride_y_0*ctr_0 + _stride_y_1*ctr_1 + _stride_y_2*ctr_2];" in code
