# (C) 2024 Irreducible Inc.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, Self, TypeVar

R = TypeVar("R")
RR = TypeVar("RR")


@dataclass(frozen=True)
class FiniteFieldElem(Generic[R]):
    """A finite field element.

    This class cannot be instantiated directly. Instead, each particular field implementation should subclass this
    and set the field class variable to an instance of FiniteField. Then the class can be instantiated with values
    of the appropriate representation.
    """

    value: R
    field: ClassVar[FiniteField]

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.field.add(self.value, other.value))

    def __mul__(self, other: Self) -> Self:
        return self.__class__(self.field.multiply(self.value, other.value))

    def __sub__(self, other: Self) -> Self:
        return self.__class__(self.field.subtract(self.value, other.value))

    def __truediv__(self, other: Self) -> Self:
        return self.__class__(self.field.divide(self.value, other.value))

    def __neg__(self) -> Self:
        return self.__class__(self.field.negate(self.value))

    def inverse(self) -> Self:
        return self.__class__(self.field.inverse(self.value))

    def square(self) -> Self:
        return self.__class__(self.field.square(self.value))

    def __pow__(self, exponent: int) -> Self:
        return self.__class__(self.field.pow(self.value, exponent))

    def __eq__(self, other) -> bool:
        return bool(self.value == other.value)

    def __neq__(self, other) -> bool:
        return bool(self.value != other.value)

    def __str__(self) -> str:
        return self.field.format_str(self.value)

    def __repr__(self) -> str:
        return self.field.format_repr(self.value)

    def __bytes__(self) -> bytes:
        return self.field.to_bytes(self.value)

    def is_zero(self) -> bool:
        return bool(self.value == self.field.zero())

    def __bool__(self) -> bool:
        return not self.is_zero()

    @classmethod
    def convert_from(cls, elem: FiniteFieldElem[RR]) -> Self:
        return cls(elem.field.convert_repr(elem.value, cls.field))

    @classmethod
    def zero(cls) -> Self:
        return cls(cls.field.zero())

    @classmethod
    def one(cls) -> Self:
        return cls(cls.field.one())

    @classmethod
    def random(cls) -> Self:
        return cls(cls.field.random())

    @classmethod
    def from_int(cls, val: int) -> Self:
        return cls(cls.field.from_int(val))

    @classmethod
    def from_bytes(cls, serialized: bytes) -> Self:
        return cls(cls.field.from_bytes(serialized))


class FiniteField(ABC, Generic[R]):
    """A finite field implementation.

    All finite fields have order p^n, where p is a prime number. p is the field characteristic and n is the degree
    of the field extension of GF(p^n) over the base field GF(p). An instance of FiniteField encapsulates the
    representation of field elements and the logic for all basic field operations: addition, negation, multiplication,
    and inversion. Two finite fields are isomorphic if the characteristic and dimension are the same, even if they
    have different representations of elements and algorithmic logic of operations.
    """

    @property
    @abstractmethod
    def characteristic(self) -> int:
        """The field characteristic, ie. the order of the base field."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimension of the field as a vector space over its base field."""
        pass

    @property
    def degree(self) -> int:
        """The degree of the field as an extension over its base field.

        Alias of dimension property.
        """
        return self.dimension

    def zero(self) -> R:
        return self.from_int(0)

    def one(self) -> R:
        return self.from_int(1)

    @abstractmethod
    def random(self) -> R:
        pass

    @abstractmethod
    def add(self, left: R, right: R) -> R:
        pass

    @abstractmethod
    def subtract(self, left: R, right: R) -> R:
        pass

    @abstractmethod
    def negate(self, operand: R) -> R:
        pass

    @abstractmethod
    def multiply(self, left: R, right: R) -> R:
        pass

    def square(self, operand: R) -> R:
        return self.multiply(operand, operand)

    def pow(self, base: R, exponent: int) -> R:
        acc = self.one()
        val = base

        while exponent:
            if exponent % 2:
                acc = self.multiply(acc, val)
            val = self.square(val)
            exponent >>= 1

        return acc

    @abstractmethod
    def inverse(self, operand: R) -> R:
        pass

    def divide(self, left: R, right: R) -> R:
        return self.multiply(left, self.inverse(right))

    @abstractmethod
    def format_str(self, elem: R) -> str:
        pass

    @abstractmethod
    def format_repr(self, elem: R) -> str:
        pass

    @abstractmethod
    def to_bytes(self, elem: R) -> bytes:
        pass

    @abstractmethod
    def from_bytes(self, serialized: bytes) -> R:
        pass

    @property
    @abstractmethod
    def bytes_len(self) -> int:
        pass

    @abstractmethod
    def from_int(self, val: int) -> R:
        """Creates a field element from an integer.

        The integer argument will be automatically converted to val % p, where p is the field's prime characteristic.
        """
        pass

    def is_isomorphic(self, field: FiniteField[RR]) -> bool:
        """Returns whether the characteristic and dimension of the fields are the same."""
        return field.characteristic == self.characteristic and field.dimension == self.dimension

    @abstractmethod
    def convert_repr(self, elem: R, field: FiniteField[RR]) -> RR:
        """Converts an element to a different field representation."""
        pass
