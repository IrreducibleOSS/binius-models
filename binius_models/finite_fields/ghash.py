from typing import ClassVar, TypeVar

import galois

from .finite_field import FiniteField, FiniteFieldElem

RR = TypeVar("RR")


class GHASHField(FiniteField[int]):
    """GHASH field: GF(2^128) with irreducible polynomial x^128 + x^7 + x^2 + x + 1"""

    def __init__(self):
        self._galois_field = galois.GF(2**128, irreducible_poly="x^128 + x^7 + x^2 + x + 1")

    @property
    def characteristic(self) -> int:
        return 2

    @property
    def dimension(self) -> int:
        return 128

    def random(self) -> int:
        return int(self._galois_field.Random())

    def add(self, left: int, right: int) -> int:
        return int(self._galois_field(left) + self._galois_field(right))

    def subtract(self, left: int, right: int) -> int:
        return int(self._galois_field(left) - self._galois_field(right))

    def negate(self, operand: int) -> int:
        return operand  # In GF(2^n), negation is identity

    def multiply(self, left: int, right: int) -> int:
        return int(self._galois_field(left) * self._galois_field(right))

    def inverse(self, operand: int) -> int:
        if operand == 0:
            raise ValueError("cannot invert zero")
        return int(self._galois_field(operand) ** -1)

    def format_str(self, elem: int) -> str:
        return f"0x{elem:032x}"

    def format_repr(self, elem: int) -> str:
        return f"GHASH(0x{elem:032x})"

    def to_bytes(self, elem: int) -> bytes:
        return elem.to_bytes(16, byteorder="big")

    def from_bytes(self, serialized: bytes) -> int:
        if len(serialized) != 16:
            raise ValueError("GHASH field elements must be 16 bytes")
        return int.from_bytes(serialized, byteorder="big")

    @property
    def bytes_len(self) -> int:
        return 16

    def from_int(self, val: int) -> int:
        return val & ((1 << 128) - 1)

    def convert_repr(self, elem: int, field: FiniteField[RR]) -> RR:
        if not self.is_isomorphic(field):
            raise ValueError("cannot convert to non-isomorphic field")
        return field.from_int(elem)


class GHASHElem(FiniteFieldElem[int]):
    field: ClassVar[GHASHField] = GHASHField()
