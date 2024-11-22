# (C) 2024 Irreducible Inc.

from __future__ import annotations

from abc import ABC, abstractmethod
from random import randrange
from typing import ClassVar, Self, TypeVar

from .finite_field import FiniteField, FiniteFieldElem

R = TypeVar("R")
RR = TypeVar("RR")


class PrimeFieldElem(FiniteFieldElem[R]):
    field: ClassVar[PrimeField]

    @classmethod
    def max(cls) -> Self:
        return cls(cls.field.max())

    def to_int(self) -> int:
        return self.field.to_int(self.value)

    def __int__(self) -> int:
        return self.to_int()


class PrimeField(FiniteField[R], ABC):
    """A subclass of FiniteField representing fields with prime order by a single integer."""

    def __init__(self, prime: int):
        self.p = prime
        self.bitlen = self.p.bit_length()
        hexlen = (self.bitlen + 3) // 4
        self.fmt = f"{{:#0{hexlen + 2:d}x}}"

    @property
    def characteristic(self) -> int:
        return self.p

    @property
    def dimension(self) -> int:
        return 1

    def max(self) -> R:
        return self.from_int(self.p - 1)

    @abstractmethod
    def to_int(self, elem: R) -> int:
        """Converts from a field element to an integer in the range [0, p)"""
        pass

    def format_repr(self, elem: R) -> str:
        return self.fmt.format(self.to_int(elem))

    @property
    def prime(self) -> int:
        return self.p

    def convert_repr(self, elem: R, field: FiniteField[RR]) -> RR:
        if not self.is_isomorphic(field):
            raise ValueError("cannot convert to non-isomorphic field")
        return field.from_int(self.to_int(elem))


class PrimeFieldIntRepr(PrimeField[int], ABC):
    def random(self) -> int:
        return randrange(0, self.prime)

    def add(self, left: int, right: int) -> int:
        return (left + right) % self.prime

    def subtract(self, left: int, right: int) -> int:
        return (left - right) % self.prime

    def negate(self, operand: int) -> int:
        return -operand % self.prime

    def format_str(self, elem: int) -> str:
        return self.fmt.format(elem)

    def to_bytes(self, elem: int) -> bytes:
        return elem.to_bytes(self.bytes_len, byteorder="little")

    def from_bytes(self, serialized: bytes) -> int:
        if len(serialized) != self.bytes_len:
            raise ValueError(f"serialized element must be {self.bytes_len} bytes")
        return int.from_bytes(serialized, byteorder="little")

    @property
    def bytes_len(self) -> int:
        return (self.bitlen + 7) // 8


class PrimeFieldNative(PrimeFieldIntRepr):
    def multiply(self, left: int, right: int) -> int:
        return (left * right) % self.p

    def pow(self, base: int, exponent: int) -> int:
        return pow(base, exponent, self.p)

    def inverse(self, operand: int) -> int:
        assert not operand == self.zero(), "divide by zero"
        return pow(operand, -1, self.p)

    def from_int(self, val: int) -> int:
        return val % self.p

    def to_int(self, elem: int) -> int:
        return elem
