import pystencils as ps


def test_create_kernel_config():
    c = ps.CreateKernelConfig()
    assert c.backend == ps.Backend.C
    assert c.target == ps.Target.CPU

    c = ps.CreateKernelConfig(target=ps.Target.GPU)
    assert c.backend == ps.Backend.CUDA

    c = ps.CreateKernelConfig(target=ps.Target.OPENCL)
    assert c.backend == ps.Backend.OPENCL

    c = ps.CreateKernelConfig(backend=ps.Backend.CUDA)
    assert c.target == ps.Target.CPU
    assert c.backend == ps.Backend.CUDA
