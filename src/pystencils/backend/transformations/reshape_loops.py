from typing import Sequence

from ..kernelcreation import KernelCreationContext, Typifier
from ..kernelcreation.ast_factory import AstFactory, IndexParsable

from ..ast.structural import PsLoop, PsBlock, PsConditional, PsDeclaration
from ..ast.expressions import PsExpression, PsConstantExpr, PsGe, PsLt
from ..constants import PsConstant

from .canonical_clone import CanonicalClone, CloneContext
from .eliminate_constants import EliminateConstants


class ReshapeLoops:
    """Various transformations for reshaping loop nests."""

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self._typify = Typifier(ctx)
        self._factory = AstFactory(ctx)
        self._canon_clone = CanonicalClone(ctx)
        self._elim_constants = EliminateConstants(ctx)

    def peel_loop_front(
        self, loop: PsLoop, num_iterations: int, omit_range_check: bool = False
    ) -> tuple[Sequence[PsBlock], PsLoop]:
        """Peel off iterations from the front of a loop.

        Removes ``num_iterations`` from the front of the given loop and returns them as a sequence of
        independent blocks.

        Args:
            loop: The loop node from which to peel iterations
            num_iterations: The number of iterations to peel off
            omit_range_check: If set to `True`, assume that the peeled-off iterations will always
              be executed, and omit their enclosing conditional.

        Returns:
            Tuple containing the peeled-off iterations as a sequence of blocks,
            and the remaining loop.
        """

        peeled_iters: list[PsBlock] = []

        for i in range(num_iterations):
            cc = CloneContext(self._ctx)
            cc.symbol_decl(loop.counter.symbol)
            peeled_ctr = self._factory.parse_index(
                cc.get_replacement(loop.counter.symbol)
            )
            peeled_idx = self._elim_constants(
                self._typify(loop.start + PsExpression.make(PsConstant(i)) * loop.step)
            )

            counter_decl = PsDeclaration(peeled_ctr, peeled_idx)
            peeled_block = self._canon_clone.visit(loop.body, cc)

            if omit_range_check:
                peeled_block.statements = [counter_decl] + peeled_block.statements
            else:
                iter_condition = PsLt(peeled_ctr, loop.stop)
                peeled_block.statements = [
                    counter_decl,
                    PsConditional(iter_condition, PsBlock(peeled_block.statements)),
                ]

            peeled_iters.append(peeled_block)

        loop.start = self._elim_constants(
            self._typify(
                loop.start + PsExpression.make(PsConstant(num_iterations)) * loop.step
            )
        )

        return peeled_iters, loop

    def peel_loop_back(
        self, loop: PsLoop, num_iterations: int, omit_range_check: bool = False
    ) -> tuple[PsLoop, Sequence[PsBlock]]:
        """Peel off iterations from the back of a loop.

        Removes ``num_iterations`` from the back of the given loop and returns them as a sequence of
        independent blocks.

        Args:
            loop: The loop node from which to peel iterations
            num_iterations: The number of iterations to peel off
            omit_range_check: If set to `True`, assume that the peeled-off iterations will always
              be executed, and omit their enclosing conditional.

        Returns:
            Tuple containing the modified loop and the peeled-off iterations (sequence of blocks).
        """

        if not (
            isinstance(loop.step, PsConstantExpr) and loop.step.constant.value == 1
        ):
            raise NotImplementedError(
                "Peeling iterations from the back of loops is only implemented"
                "for loops with unit step. Implementation is deferred until"
                "loop range canonicalization is available (also needed for the"
                "vectorizer)."
            )

        peeled_iters: list[PsBlock] = []

        for i in range(num_iterations)[::-1]:
            cc = CloneContext(self._ctx)
            cc.symbol_decl(loop.counter.symbol)
            peeled_ctr = self._factory.parse_index(
                cc.get_replacement(loop.counter.symbol)
            )
            peeled_idx = self._typify(loop.stop - PsExpression.make(PsConstant(i + 1)))

            counter_decl = PsDeclaration(peeled_ctr, peeled_idx)
            peeled_block = self._canon_clone.visit(loop.body, cc)

            if omit_range_check:
                peeled_block.statements = [counter_decl] + peeled_block.statements
            else:
                iter_condition = PsGe(peeled_ctr, loop.start)
                peeled_block.statements = [
                    counter_decl,
                    PsConditional(iter_condition, PsBlock(peeled_block.statements)),
                ]

            peeled_iters.append(peeled_block)

        loop.stop = self._elim_constants(
            self._typify(loop.stop - PsExpression.make(PsConstant(num_iterations)))
        )

        return loop, peeled_iters

    def cut_loop(
        self, loop: PsLoop, cutting_points: Sequence[IndexParsable]
    ) -> Sequence[PsLoop | PsBlock]:
        """Cut a loop at the given cutting points.

        Cut the given loop at the iterations specified by the given cutting points,
        producing ``n`` new subtrees representing the iterations
        ``(loop.start:cutting_points[0]), (cutting_points[0]:cutting_points[1]), ..., (cutting_points[-1]:loop.stop)``.

        Resulting subtrees representing zero iterations are dropped; subtrees representing exactly one iteration are
        returned without the trivial loop structure.

        Currently, `cut_loop` performs no checks to ensure that the given cutting points are in fact inside
        the loop's iteration range.

        Returns:
            Sequence of ``n`` subtrees representing the respective iteration ranges
        """

        if not (
            isinstance(loop.step, PsConstantExpr) and loop.step.constant.value == 1
        ):
            raise NotImplementedError(
                "Loop cutting for loops with step != 1 is not implemented"
            )

        result: list[PsLoop | PsBlock] = []
        new_start = loop.start
        cutting_points = [self._factory.parse_index(idx) for idx in cutting_points] + [
            loop.stop
        ]

        for new_end in cutting_points:
            if new_end.structurally_equal(new_start):
                continue

            num_iters = self._elim_constants(self._typify(new_end - new_start))
            skip = False

            if isinstance(num_iters, PsConstantExpr):
                if num_iters.constant.value == 0:
                    skip = True
                elif num_iters.constant.value == 1:
                    skip = True
                    cc = CloneContext(self._ctx)
                    cc.symbol_decl(loop.counter.symbol)
                    local_counter = self._factory.parse_index(
                        cc.get_replacement(loop.counter.symbol)
                    )
                    ctr_decl = PsDeclaration(
                        local_counter,
                        new_start,
                    )
                    cloned_body = self._canon_clone.visit(loop.body, cc)
                    cloned_body.statements = [ctr_decl] + cloned_body.statements
                    result.append(cloned_body)

            if not skip:
                loop_clone = self._canon_clone(loop)
                loop_clone.start = new_start.clone()
                loop_clone.stop = new_end.clone()
                result.append(loop_clone)

            new_start = new_end

        return result
