from typing import Any


def failing_cast(target: type | tuple[type, ...], obj: Any) -> Any:
    if not isinstance(obj, target):
        raise TypeError(f"Casting {obj} to {target} failed.")
    return obj
