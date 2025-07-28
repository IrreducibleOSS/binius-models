from typing import TypeVar

from .finite_field import FiniteFieldElem

F = TypeVar("F", bound=FiniteFieldElem)


def tensor_expand(field: type[F], x: list[F], vars: int) -> list[F]:
    result = [field.one()] * (1 << vars)
    for i in range(vars):
        for j in range(1 << i):
            result[1 << i | j] = result[j] * x[i]
            result[j] -= result[1 << i | j]
    return result
