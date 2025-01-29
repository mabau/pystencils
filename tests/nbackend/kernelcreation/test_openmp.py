import pytest
from pystencils import (
    fields,
    Assignment,
    create_kernel,
    CreateKernelConfig,
    Target,
)

from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.structural import PsLoop, PsPragma


@pytest.mark.parametrize("nesting_depth", range(3))
@pytest.mark.parametrize("schedule", ["static", "static,16", "dynamic", "auto"])
@pytest.mark.parametrize("collapse", [None, 1, 2])
@pytest.mark.parametrize("omit_parallel_construct", range(3))
def test_openmp(nesting_depth, schedule, collapse, omit_parallel_construct):
    f, g = fields("f, g: [3D]")
    asm = Assignment(f.center(0), g.center(0))

    gen_config = CreateKernelConfig(target=Target.CPU)
    gen_config.cpu.openmp.enable = True
    gen_config.cpu.openmp.nesting_depth = nesting_depth
    gen_config.cpu.openmp.schedule = schedule
    gen_config.cpu.openmp.collapse = collapse
    gen_config.cpu.openmp.omit_parallel_construct = omit_parallel_construct

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

    expected_tokens = {"omp", "for", f"schedule({schedule})"}
    if not omit_parallel_construct:
        expected_tokens.add("parallel")
    if collapse is not None:
        expected_tokens.add(f"collapse({collapse})")

    assert tokens == expected_tokens
