import pytest
from pystencils import (
    fields,
    Assignment,
    create_kernel,
    CreateKernelConfig,
    CpuOptimConfig,
    OpenMpConfig,
    Target,
)

from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.structural import PsLoop, PsPragma


@pytest.mark.parametrize("nesting_depth", range(3))
@pytest.mark.parametrize("schedule", ["static", "static,16", "dynamic", "auto"])
@pytest.mark.parametrize("collapse", range(3))
@pytest.mark.parametrize("omit_parallel_construct", range(3))
def test_openmp(nesting_depth, schedule, collapse, omit_parallel_construct):
    f, g = fields("f, g: [3D]")
    asm = Assignment(f.center(0), g.center(0))

    omp = OpenMpConfig(
        nesting_depth=nesting_depth,
        schedule=schedule,
        collapse=collapse,
        omit_parallel_construct=omit_parallel_construct,
    )
    gen_config = CreateKernelConfig(
        target=Target.CPU, cpu_optim=CpuOptimConfig(openmp=omp)
    )

    kernel = create_kernel(asm, gen_config)
    ast = kernel.body

    def find_omp_pragma(ast) -> PsPragma:
        num_loops = 0
        generator = dfs_preorder(ast)
        for node in generator:
            match node:
                case PsLoop():
                    num_loops += 1
                case PsPragma():
                    loop = next(generator)
                    assert isinstance(loop, PsLoop)
                    assert num_loops == nesting_depth
                    return node

        pytest.fail("No OpenMP pragma found")

    pragma = find_omp_pragma(ast)
    tokens = set(pragma.text.split())

    expected_tokens = {"omp", "for", f"schedule({omp.schedule})"}
    if not omp.omit_parallel_construct:
        expected_tokens.add("parallel")
    if omp.collapse > 0:
        expected_tokens.add(f"collapse({omp.collapse})")

    assert tokens == expected_tokens
