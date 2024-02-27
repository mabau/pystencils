from __future__ import annotations
from typing import Sequence
from abc import ABC, abstractmethod


class PsAstNode(ABC):
    """Base class for all nodes in the pystencils AST.

    This base class provides a common interface to inspect and update the AST's branching structure.
    The two methods `get_children` and `set_child` must be implemented by each subclass.
    Subclasses are also responsible for doing the necessary type checks if they place restrictions on
    the types of their children.
    """

    @property
    def children(self) -> Sequence[PsAstNode]:
        return self.get_children()

    @children.setter
    def children(self, cs: Sequence[PsAstNode]):
        for i, c in enumerate(cs):
            self.set_child(i, c)

    @abstractmethod
    def get_children(self) -> tuple[PsAstNode, ...]:
        pass

    @abstractmethod
    def set_child(self, idx: int, c: PsAstNode):
        pass

    def structurally_equal(self, other: PsAstNode) -> bool:
        """Check two ASTs for structural equality."""
        return (
            (type(self) is type(other))
            and len(self.children) == len(other.children)
            and all(
                c1.structurally_equal(c2)
                for c1, c2 in zip(self.children, other.children)
            )
        )


class PsLeafMixIn(ABC):
    """Mix-in for AST leaves."""

    def get_children(self) -> tuple[PsAstNode, ...]:
        return ()

    def set_child(self, idx: int, c: PsAstNode):
        raise IndexError("Child index out of bounds: Leaf nodes have no children.")

    @abstractmethod
    def structurally_equal(self, other: PsAstNode) -> bool:
        pass
