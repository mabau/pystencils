from __future__ import annotations

from functools import wraps

from typing import Callable
from types import MethodType

from .nodes import PsAstNode


class VisitorDispatcher:
    def __init__(self, wrapped_method):
        self._dispatch_dict = {}
        self._wrapped_method = wrapped_method

    def case(self, node_type: type):
        """Decorator for visitor's methods"""

        def decorate(handler: Callable):
            if node_type in self._dispatch_dict:
                raise ValueError(f"Duplicate visitor case {node_type}")
            self._dispatch_dict[node_type] = handler
            return handler

        return decorate

    def __call__(self, instance, node: PsAstNode, *args, **kwargs):
        for cls in node.__class__.mro():
            if cls in self._dispatch_dict:
                return self._dispatch_dict[cls](instance, node, *args, **kwargs)

        return self._wrapped_method(instance, node, *args, **kwargs)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return MethodType(self, obj)


def ast_visitor(method):
    return wraps(method)(VisitorDispatcher(method))
