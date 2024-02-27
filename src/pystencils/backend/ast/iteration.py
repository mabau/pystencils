from typing import Callable, Generator

from .structural import PsAstNode


def dfs_preorder(
    node: PsAstNode, yield_pred: Callable[[PsAstNode], bool] = lambda _: True
) -> Generator[PsAstNode, None, None]:
    """Pre-Order depth-first traversal of an abstract syntax tree.

    Args:
        node: The tree's root node
        yield_pred: Filter predicate; a node is only yielded to the caller if `yield_pred(node)` returns True
    """
    if yield_pred(node):
        yield node

    for c in node.children:
        yield from dfs_preorder(c, yield_pred)


def dfs_postorder(
    node: PsAstNode, yield_pred: Callable[[PsAstNode], bool] = lambda _: True
) -> Generator[PsAstNode, None, None]:
    """Post-Order depth-first traversal of an abstract syntax tree.

    Args:
        node: The tree's root node
        yield_pred: Filter predicate; a node is only yielded to the caller if `yield_pred(node)` returns True
    """
    for c in node.children:
        yield from dfs_postorder(c, yield_pred)

    if yield_pred(node):
        yield node
