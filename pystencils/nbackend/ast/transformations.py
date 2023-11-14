from abc import ABC

from typing import Dict

from pymbolic.primitives import Expression
from pymbolic.mapper.substitutor import CachedSubstitutionMapper

from ..typed_expressions import PsTypedSymbol
from .dispatcher import ast_visitor
from .nodes import PsAstNode, PsAssignment, PsLoop, PsExpression


class PsAstTransformer(ABC):
    def transform_children(self, node: PsAstNode, *args, **kwargs):
        node.children = [self.visit(c, *args, **kwargs) for c in node.children]

    @ast_visitor
    def visit(self, node, *args, **kwargs):
        self.transform_children(node, *args, **kwargs)
        return node


class PsSymbolsSubstitutor(PsAstTransformer):
    def __init__(self, subs_dict: Dict[PsTypedSymbol, Expression]):
        self._subs_dict = subs_dict
        self._mapper = CachedSubstitutionMapper(lambda s: self._subs_dict.get(s, None))

    def subs(self, node: PsAstNode):
        return self.visit(node)

    visit = PsAstTransformer.visit

    @visit.case(PsAssignment)
    def assignment(self, asm: PsAssignment):
        lhs_expr = asm.lhs.expression
        if isinstance(lhs_expr, PsTypedSymbol) and lhs_expr in self._subs_dict:
            raise ValueError(f"Cannot substitute symbol {lhs_expr} that occurs on a left-hand side of an assignment.")
        self.transform_children(asm)
        return asm

    @visit.case(PsLoop)
    def loop(self, loop: PsLoop):
        if loop.counter.expression in self._subs_dict:
            raise ValueError(f"Cannot substitute symbol {loop.counter.expression} that is defined as a loop counter.")
        self.transform_children(loop)
        return loop

    @visit.case(PsExpression)
    def expression(self, expr_node: PsExpression):
        self._mapper(expr_node.expression)


def ast_subs(node: PsAstNode, subs_dict: dict):
    return PsSymbolsSubstitutor(subs_dict).subs(node)
