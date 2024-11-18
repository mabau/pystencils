from typing import Callable, Generator

from .structural import PsAstNode


def dfs_preorder(
    node: PsAstNode, filter_pred: Callable[[PsAstNode], bool] = lambda _: True
) -> Generator[PsAstNode, None, None]:
    """Pre-Order depth-first traversal of an abstract syntax tree.

    Args:
        node: The tree's root node
        filter_pred: Filter predicate; a node is only returned to the caller if ``yield_pred(node)`` returns True
    """
    if filter_pred(node):
        yield node

    for c in node.children:
        yield from dfs_preorder(c, filter_pred)


def dfs_postorder(
    node: PsAstNode, filter_pred: Callable[[PsAstNode], bool] = lambda _: True
) -> Generator[PsAstNode, None, None]:
    """Post-Order depth-first traversal of an abstract syntax tree.

    Args:
        node: The tree's root node
        filter_pred: Filter predicate; a node is only returned to the caller if ``yield_pred(node)`` returns True
    """
    for c in node.children:
        yield from dfs_postorder(c, filter_pred)

    if filter_pred(node):
        yield node
