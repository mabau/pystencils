import os
from time import perf_counter
import subprocess
from tempfile import TemporaryDirectory

from pystencils import create_data_handling
from pystencils.backends.cbackend import CBackend
from jinja2 import Environment, FileSystemLoader
from pystencils.backends.cbackend import generate_c

script_path = os.path.dirname(os.path.realpath(__file__))
PAXX_ROOT = '/local/bauer/code/pacxx/install'
DEFAULT_PAXX_COMPILE_OPTIONS = ('-Ofast', '-march=native')


def generate_benchmark_code(target_file, kernel_ast, target):
    assert target in ('cpu', 'gpu')
    assert hasattr(kernel_ast, 'indexing'), "AST has to be a CUDA kernel in order to create a PACXX kernel from it"
    backend = CBackend()

    function_body = kernel_ast.body
    f_sizes = {f.shape[-1] for f in kernel_ast.fields_accessed}
    assert len(f_sizes) == 1

    env = Environment(loader=FileSystemLoader(script_path))
    result = env.get_template("benchmark_template.cpp").render(f_size=f_sizes.pop(),
                                                               code=backend(function_body),
                                                               target=target)

    with open(target_file, 'w') as f:
        f.write(result)


def pacxx_compile(source, executable, options=DEFAULT_PAXX_COMPILE_OPTIONS):
    command = ['pacxx++', *options, source, '-o', executable, ]
    env = os.environ.copy()
    env['PATH'] = "{}:{}".format(env.get('PATH', ''), os.path.join(PAXX_ROOT, 'bin'))
    env['LD_LIBRARY_PATH'] = "{}:{}".format(env.get('LD_LIBRARY_PATH', ''), os.path.join(PAXX_ROOT, 'lib'))
    try:
        subprocess.check_output(command, env=env, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(" ".join(command))
        print(e.output.decode('utf8'))
        raise e


def run_paxx_benchmark(executable, domain_size, iterations):
    assert len(domain_size) == 3
    arguments = [executable, *domain_size, iterations]
    arguments = [str(e) for e in arguments]
    output = subprocess.check_output(arguments)
    return float(output) / iterations


def paxx_benchmark(ast, domain_size, iterations, target='cpu', compile_options=DEFAULT_PAXX_COMPILE_OPTIONS):
    """Generates,  compiles and runs the kernel with PAXX

    Args:
        ast: pystencils AST object (has to be generated for CUDA, even when run on CPU with pacxx)
        domain_size: x, y, z extent of spatial domain
        iterations: number of outer iterations
        target: either 'cpu' or 'gpu' to specify where pacxx should run the kernel
        compile_options: compile options for pacxx

    Returns:
        seconds for one outer iteration
    """
    with TemporaryDirectory() as base_dir:
        code = os.path.join(base_dir, 'code.cpp')
        executable = os.path.join(base_dir, 'bench')
        generate_benchmark_code(code, ast, target)
        pacxx_compile(code, executable, compile_options)
        time_per_iteration = run_paxx_benchmark(executable, domain_size, iterations)
    return time_per_iteration


def lbm_performance_compare(domain_size, iterations, **lb_params):
    """Runs benchmark with pacxx and with normal pystencils backends.

    Args:
        domain_size: 3-tuple with size of spatial domain
        iterations: number of outer iterations
        **lb_params: parameters passed to lbmpy to choose lattice Boltzmann algorithm & optimization options

    Returns:
        dictionary with measurements of time per iteration for different backends
    """
    import pycuda.driver as drv

    from lbmpy.creationfunctions import create_lb_ast
    if 'optimization' not in lb_params:
        lb_params['optimization'] = {}

    lb_params['optimization']['target'] = 'cpu'
    cpu_ast = create_lb_ast(**lb_params)
    lb_params['optimization']['target'] = 'gpu'
    gpu_ast = create_lb_ast(**lb_params)

    # print kernel code of CPU or GPU version - just for comparison, files are not used
    with open("pystencils_cpu_code.c", 'w') as f:
        print(generate_c(cpu_ast), file=f)
    with open("pystencils_gpu_code.cu", 'w') as f:
        print(generate_c(gpu_ast), file=f)

    cpu_kernel = cpu_ast.compile()
    gpu_kernel = gpu_ast.compile()
    f_sizes = {f.shape[-1] for f in cpu_ast.fields_accessed}
    assert len(f_sizes) == 1
    f_size = f_sizes.pop()

    dh = create_data_handling(domain_size, default_target='gpu', default_layout='fzyx')
    dh.add_array('src', values_per_cell=f_size)
    dh.add_array('dst', values_per_cell=f_size)
    dh.fill('src', 0)
    dh.fill('dst', 0)

    # to keep it simple we run outer loop directly from Python
    # make domain size large enough, otherwise we measure the python call overhead
    def run_benchmark(kernel):
        dh.all_to_gpu()
        for i in range(10):  # warmup
            dh.run_kernel(kernel)
        drv.Context.synchronize()
        start = perf_counter()
        for i in range(iterations):
            dh.run_kernel(kernel)
        drv.Context.synchronize()
        return (perf_counter() - start) / iterations

    return {
        'pystencils_cpu': run_benchmark(cpu_kernel),
        'pystencils_gpu': run_benchmark(gpu_kernel),
        'pacxx_cpu': paxx_benchmark(gpu_ast, domain_size, iterations, target='cpu'),
        'pacxx_gpu': paxx_benchmark(gpu_ast, domain_size, iterations, target='gpu'),
    }


if __name__ == '__main__':
    no_opt = {
        'openmp': 8,  # number of threads - pacxx uses also HT cores
        'split': False,
        'vectorization': False,
        'gpu_indexing_params': {'block_size': (64, 8, 1)},
    }
    only_vectorization = {
        'openmp': 4,
        'split': False,
        'gpu_indexing_params': {'block_size': (64, 8, 1)},
        'vectorization': {'instruction_set': 'avx',
                          'assume_inner_stride_one': True,
                          'nontemporal': False},
    }
    best = {
        'openmp': 4,
        'split': True,
        'gpu_indexing_params': {'block_size': (64, 8, 1)},
        'vectorization': {'instruction_set': 'avx',
                          'assume_inner_stride_one': True,
                          'nontemporal': True}
    }
    res = lbm_performance_compare(stencil='D3Q19', relaxation_rate=1.8, compressible=False,
                                  domain_size=(512, 128, 32), iterations=500,
                                  optimization=only_vectorization)
    cpu_speedup = ((res['pacxx_cpu'] / res['pystencils_cpu']) - 1) * 100
    gpu_speedup = ((res['pacxx_gpu'] / res['pystencils_gpu']) - 1) * 100
    print("Time for one kernel call [s]")
    for config_name, time in res.items():
        print("  {0: <16}: {1}".format(config_name, time))

    print("CPU {:.02f}%   GPU {:.02f}%".format(cpu_speedup, gpu_speedup))
