import math
import os
import time

import numpy as np
import sympy as sp
from git import Repo
from influxdb import InfluxDBClient
from kerncraft.machinemodel import MachineModel
from kerncraft.models import ECM, Benchmark, Roofline, RooflineIACA
from kerncraft.prefixedunit import PrefixedUnit

from pystencils import Assignment, Field, create_kernel
from pystencils.kerncraft_coupling import KerncraftParameters, PyStencilsKerncraftKernel


def output_benchmark(analysis):
    output = {}
    keys = ['Runtime (per repetition) [s]', 'Iterations per repetition',
            'Runtime (per cacheline update) [cy/CL]', 'MEM volume (per repetition) [B]',
            'Performance [MFLOP/s]', 'Performance [MLUP/s]', 'Performance [MIt/s]', 'MEM BW [MByte/s]']
    copies = {key: analysis[key] for key in keys}
    output.update(copies)

    for cache, metrics in analysis['data transfers'].items():
        for metric_name, metric_value in metrics.items():
            fixed = metric_value.with_prefix('')
            output[cache + ' ' + metric_name + ' ' + fixed.prefix + fixed.unit] = fixed.value

    for level, value in analysis['ECM'].items():
        output['Phenomenological ECM ' + level + ' cy/CL'] = value
    return output


def output_ecm(analysis):
    output = {}
    keys = ['T_nOL', 'T_OL', 'cl throughput', 'uops']
    copies = {key: analysis[key] for key in keys}
    output.update(copies)

    if 'memory bandwidth kernel' in analysis:
        output['memory bandwidth kernel' + analysis['memory bandwidth kernel'] + analysis['memory bandwidth'].prefix +
               analysis['memory bandwidth'].unit] = analysis['memory bandwidth'].value

    output['scaling cores'] = int(analysis['scaling cores']) if not math.isinf(analysis['scaling cores']) else -1

    for key, value in analysis['cycles']:
        output[key] = value
    return output


def output_roofline(analysis):
    output = {}
    keys = ['min performance']  # 'bottleneck level'
    copies = {key: analysis[key] for key in keys}
    output.update(copies)
    # TODO save bottleneck information (compute it here)

    # fixed = analysis['max_flops'].with_prefix('G')
    # output['max GFlop/s'] = fixed.value

    # if analysis['min performance'] > max_flops:
    #    # CPU bound
    #    print('CPU bound with {} cores(s)'.format(self._args.cores), file=output_file)
    #    print('{!s} due to CPU max. FLOP/s'.format(max_flops), file=output_file)
    # else:
    # Memory bound
    bottleneck = analysis['mem bottlenecks'][analysis['bottleneck level']]
    output['bottleneck GFlop/s'] = bottleneck['performance'].with_prefix('G').value
    output['bottleneck level'] = bottleneck['level']
    output['bottleneck bw kernel'] = bottleneck['bw kernel']
    output['bottleneck arithmetic intensity'] = bottleneck['arithmetic intensity']

    for i, level in enumerate(analysis['mem bottlenecks']):
        if level is None:
            continue
        for key, value in level.items():
            if isinstance(value, PrefixedUnit):
                fixed = value.with_prefix('G')
                output['level ' + str(i) + ' ' + key + ' [' + fixed.prefix + fixed.unit + ']'] = 'inf' if isinstance(
                    fixed.value, float) and math.isinf(fixed.value) else fixed.value
            else:
                output['level ' + str(i) + ' ' + key] = 'inf' if isinstance(value, float) and math.isinf(
                    value) else value
    return output


def output_roofline_iaca(analysis):
    output = {}
    keys = ['min performance']  # 'bottleneck level'
    copies = {key: analysis[key] for key in keys}
    # output.update(copies)
    # TODO save bottleneck information (compute it here)

    # fixed = analysis['max_flops'].with_prefix('G')
    # output['max GFlop/s'] = fixed.value

    # if analysis['min performance'] > max_flops:
    #    # CPU bound
    #    print('CPU bound with {} cores(s)'.format(self._args.cores), file=output_file)
    #    print('{!s} due to CPU max. FLOP/s'.format(max_flops), file=output_file)
    # else:
    # Memory bound
    bottleneck = analysis['mem bottlenecks'][analysis['bottleneck level']]
    output['bottleneck GFlop/s'] = bottleneck['performance'].with_prefix('G').value
    output['bottleneck level'] = bottleneck['level']
    output['bottleneck bw kernel'] = bottleneck['bw kernel']
    output['bottleneck arithmetic intensity'] = bottleneck['arithmetic intensity']

    for i, level in enumerate(analysis['mem bottlenecks']):
        if level is None:
            continue
        for key, value in level.items():
            if isinstance(value, PrefixedUnit):
                fixed = value.with_prefix('G')
                output['level ' + str(i) + ' ' + key + ' [' + fixed.prefix + fixed.unit + ']'] = 'inf' if isinstance(
                    fixed.value, float) and math.isinf(fixed.value) else fixed.value
            else:
                output['level ' + str(i) + ' ' + key] = 'inf' if isinstance(value, float) and math.isinf(
                    value) else value
    return output


def report_analysis(ast, models, machine, tags, fields=None):
    kernel = PyStencilsKerncraftKernel(ast, machine)
    client = InfluxDBClient('i10grafana.informatik.uni-erlangen.de', 8086, 'pystencils',
                            'roggan', 'pystencils')
    repo = Repo(search_parent_directories=True)
    commit = repo.head.commit
    point_time = int(time.time())

    for model in models:
        benchmark = model(kernel, machine, KerncraftParameters())
        benchmark.analyze()
        analysis = benchmark.results
        if model is Benchmark:
            output = output_benchmark(analysis)
        elif model is ECM:
            output = output_ecm(analysis)
        elif model is Roofline:
            output = output_roofline(analysis)
        elif model is RooflineIACA:
            output = output_roofline_iaca(analysis)
        else:
            raise ValueError('No valid model for analysis given!')

        if fields is not None:
            output.update(fields)

        output['commit'] = commit.hexsha

        json_body = [
            {
                'measurement': model.__name__,
                'tags': tags,
                'time': point_time,
                'fields': output
            }
        ]
        client.write_points(json_body, time_precision='s')


def main():
    size = [20, 200, 200]
    arr = np.zeros(size)
    a = Field.create_from_numpy_array('a', arr, index_dimensions=0)
    b = Field.create_from_numpy_array('b', arr, index_dimensions=0)
    s = sp.Symbol("s")
    rhs = a[0, -1, 0] + a[0, 1, 0] + \
          a[-1, 0, 0] + a[1, 0, 0] + \
          a[0, 0, -1] + a[0, 0, 1]

    update_rule = Assignment(b[0, 0, 0], s * rhs)
    ast = create_kernel([update_rule])
    input_folder = "./"
    machine_file_path = os.path.join(input_folder, "SkylakeSP_Gold-5122_allinclusive.yaml")
    machine = MachineModel(path_to_yaml=machine_file_path)
    tags = {
        'host': os.uname()[1],
        'project': 'pystencils',
        'kernel': 'jacobi_3D ' + str(size)
    }

    report_analysis(ast, [ECM, Roofline, RooflineIACA, Benchmark], machine, tags)


if __name__ == '__main__':
    main()
