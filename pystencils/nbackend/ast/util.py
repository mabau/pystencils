from typing import TypeVar

T = TypeVar("T")


def failing_cast(target: type, obj: T):
    if not isinstance(obj, target):
        raise TypeError(f"Casting {obj} to {target} failed.")
    return obj
