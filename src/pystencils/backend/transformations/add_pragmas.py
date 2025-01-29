from __future__ import annotations
from dataclasses import dataclass

from typing import Sequence
from collections import defaultdict

from ..kernelcreation import KernelCreationContext
from ..ast import PsAstNode
from ..ast.structural import PsBlock, PsLoop, PsPragma
from ..ast.expressions import PsExpression


__all__ = ["InsertPragmasAtLoops", "LoopPragma", "AddOpenMP"]


@dataclass
class LoopPragma:
    """A pragma that should be prepended to loops at a certain nesting depth."""

    text: str
    """The pragma text, without the ``#pragma ``"""

    loop_nesting_depth: int
    """Nesting depth of the loops the pragma should be added to. ``-1`` indicates the innermost loops."""

    def __post_init__(self):
        if self.loop_nesting_depth < -1:
            raise ValueError("Loop nesting depth must be nonnegative or -1.")


@dataclass
class Nesting:
    depth: int
    has_inner_loops: bool = False


class InsertPragmasAtLoops:
    """Insert pragmas before loops in a loop nest.

    This transformation augments the AST with pragma directives which are prepended to loops.
    The directives are annotated with the nesting depth of the loops they should be added to,
    where ``-1`` indicates the innermost loop.

    The relative order of pragmas with the (exact) same nesting depth is preserved;
    however, no guarantees are given about the relative order of pragmas inserted at ``-1``
    and at the actual depth of the innermost loop.
    """

    def __init__(
        self, ctx: KernelCreationContext, insertions: Sequence[LoopPragma]
    ) -> None:
        self._ctx = ctx
        self._insertions: dict[int, list[LoopPragma]] = defaultdict(list)
        for ins in insertions:
            self._insertions[ins.loop_nesting_depth].append(ins)

    def __call__(self, node: PsAstNode) -> PsAstNode:
        is_loop = isinstance(node, PsLoop)
        if is_loop:
            node = PsBlock([node])

        self.visit(node, Nesting(0))

        if is_loop and len(node.children) == 1:
            node = node.children[0]

        return node

    def visit(self, node: PsAstNode, nest: Nesting) -> None:
        match node:
            case PsExpression():
                return

            case PsBlock(children):
                new_children: list[PsAstNode] = []
                for c in children:
                    if isinstance(c, PsLoop):
                        nest.has_inner_loops = True
                        inner_nest = Nesting(nest.depth + 1)
                        self.visit(c.body, inner_nest)

                        if not inner_nest.has_inner_loops:
                            # c is the innermost loop
                            for pragma in self._insertions[-1]:
                                new_children.append(PsPragma(pragma.text))

                        for pragma in self._insertions[nest.depth]:
                            new_children.append(PsPragma(pragma.text))

                    new_children.append(c)
                node.children = new_children

            case other:
                for c in other.children:
                    self.visit(c, nest)


class AddOpenMP:
    """Apply OpenMP directives to loop nests.

    This transformation augments the AST with OpenMP pragmas according to the given configuration.
    """

    def __init__(
        self,
        ctx: KernelCreationContext,
        nesting_depth: int = 0,
        num_threads: int | None = None,
        schedule: str | None = None,
        collapse: int | None = None,
        omit_parallel: bool = False,
    ) -> None:
        pragma_text = "omp"

        if not omit_parallel:
            pragma_text += " parallel"

        pragma_text += " for"

        if schedule is not None:
            pragma_text += f" schedule({schedule})"

        if num_threads is not None:
            pragma_text += f" num_threads({str(num_threads)})"

        if collapse is not None:
            if collapse <= 0:
                raise ValueError(
                    f"Invalid value for OpenMP `collapse` clause: {collapse}"
                )
            pragma_text += f" collapse({str(collapse)})"

        self._insert_pragmas = InsertPragmasAtLoops(
            ctx, [LoopPragma(pragma_text, nesting_depth)]
        )

    def __call__(self, node: PsAstNode) -> PsAstNode:
        return self._insert_pragmas(node)
