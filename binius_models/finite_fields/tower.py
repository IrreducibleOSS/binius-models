from __future__ import annotations

import random
from abc import ABC
from typing import ClassVar, Self, TypeVar

from galois import GF

from binius_models.finite_fields.finite_field import FiniteField, FiniteFieldElem
from binius_models.utils.utils import bits_mask, factorize

RR = TypeVar("RR")
F = TypeVar("F", bound="BinaryTowerFieldElem")


class BinaryTowerField(FiniteField[int], ABC):
    subfield: BinaryTowerField | None

    def __init__(self, degree: int):
        self._degree = degree  # aren't self._degree and self.degree different?
        hexlen = (self.degree + 3) // 4
        self.fmt = f"{{:#0{hexlen + 2:d}x}}"

    @property
    def characteristic(self) -> int:
        return 2

    @property
    def dimension(self) -> int:
        return self._degree

    def random(self) -> int:
        return random.randrange(1 << self.dimension)

    def add(self, left: int, right: int) -> int:
        return left ^ right

    def subtract(self, left: int, right: int) -> int:
        return left ^ right

    def negate(self, operand: int) -> int:
        return operand

    def format_str(self, elem: int) -> str:
        return self.fmt.format(elem)

    def format_repr(self, elem: int) -> str:
        return self.fmt.format(elem)

    def to_bytes(self, elem: int) -> bytes:
        return elem.to_bytes(self.bytes_len, byteorder="little")

    def is_generator(self, elem: int) -> bool:
        if elem == 0:
            return False
        factors = factorize(2**self.dimension - 1)
        return not any(self.pow(elem, (2**self.dimension - 1) // factor) == self.one() for factor in factors)

    def random_multiplicative_generator(self) -> int:
        while True:
            result = self.random()
            if self.is_generator(result):
                return result

    def multiplicative_generator(self) -> int:
        """Returns the smallest viable multiplicative generator."""
        for i in range(2 ** (self.degree // 2), 2**self.degree):
            g = self.from_int(i)
            if self.is_generator(g):
                return g
        raise ValueError("no multiplicative generator found")

    def from_bytes(self, serialized: bytes) -> int:
        if len(serialized) != self.bytes_len:
            raise ValueError(f"serialized element must be {self.bytes_len} bytes")
        return int.from_bytes(serialized, byteorder="little")

    @property
    def bytes_len(self) -> int:
        return (self.dimension + 7) // 8

    def from_int(self, val: int) -> int:
        return val & 1

    def convert_repr(self, elem: int, field: FiniteField[RR]) -> RR:
        raise NotImplementedError()

    def to_subfield_tuple(self, elem: int) -> tuple[int, ...]:
        if self.subfield is None:
            raise ValueError("there is no subfield of the base field")
        m = self.subfield.degree
        return tuple(elem >> m * i & bits_mask(m) for i in range(self.degree // self.subfield.degree))

    def from_subfield_tuple(self, elem: tuple[int, ...]) -> int:
        if self.subfield is None:
            raise ValueError("there is no subfield of the base field")
        m = self.subfield.degree
        result = 0
        for i, component in enumerate(elem):
            result |= component << i * m
        return result

    def _is_valid(self, val: int) -> bool:
        return val >> self.degree == 0

    def _multiply_alpha(self, a: int) -> int:
        raise NotImplementedError()


class FASTowerField(BinaryTowerField):
    """
    The polynomial modulus for step n of the tower is Xₙ² + Xₙ + α, where α = X₀ ⋅ ⋯ ⋅ Xₙ₋₁.
    """

    subfield: FASTowerField | None
    generators = {  # class member
        1: 0x1,
        2: 0x3,
        4: 0x5,
        8: 0xCA,
        16: 0xD7AF,
        32: 0xA15F6C4D,
        64: 0x58394D0CB926D29A,
        128: 0x8CC63F6BF3D1C66A2364BAE373B784BD,
    }

    def __init__(self, height: int) -> None:
        super().__init__(1 << height)
        self.fmt = f"FASTowerField({self.fmt})"
        if height == 0:
            self.subfield = None
        else:
            self.subfield = FASTowerField(height - 1)

    def multiplicative_generator(self) -> int:
        """Returns a generator of the multiplicative group of units.

        This method is very fast, essentially a lookup of precomputed constants.

        :raises NotImplementedError: if the multiplicative generator for this field is not precomputed
        """
        if self.degree > 128:
            raise NotImplementedError
        return self.generators[self.degree]

    def multiply(self, a: int, b: int) -> int:
        # recursive tower mult; uses 2×2 Karatsuba at each step
        # https://en.wikipedia.org/wiki/Karatsuba_algorithm
        if self.subfield is None:  # base case
            return a & b  # single-bit AND gate
        a0, a1 = self.to_subfield_tuple(a)
        b0, b1 = self.to_subfield_tuple(b)
        z0 = self.subfield.multiply(a0, b0)
        z2 = self.subfield.multiply(a1, b1)
        z1 = self.subfield.multiply(a0 ^ a1, b0 ^ b1) ^ z0
        z2a = self.subfield._multiply_alpha(z2)  # this mult is by a constant, unlike the others
        # it seems possible/likely that the synthesizer would be able to optimize away some of the complexity here
        # if not, we can explicitly turn this into a matrix mult and try to make it easier
        return self.from_subfield_tuple((z0 ^ z2a, z1))

    def _multiply_alpha(self, a: int) -> int:
        if self.subfield is None:  # base case
            return a
        a0, a1 = self.to_subfield_tuple(a)
        # Here we think of b = alpha, where b0 = 0 and b1 is the alpha of the subfield
        # Since b0 = 0, we don't need Karatsuba multiplication
        z2 = self.subfield._multiply_alpha(a1)
        z1 = self.subfield._multiply_alpha(a0)
        z2a = self.subfield._multiply_alpha(z2)
        return self.from_subfield_tuple((z2a, z1 ^ z2))

    def square(self, a: int) -> int:
        if self.subfield is None:  # base case
            return a
        a0, a1 = self.to_subfield_tuple(a)
        z0 = self.subfield.square(a0)
        z2 = self.subfield.square(a1)
        z2a = self.subfield._multiply_alpha(z2)
        return self.from_subfield_tuple((z0 ^ z2a, z2))

    def inverse(self, a: int) -> int:
        # Fan and Paar. On Efficient  Inversion in  Tower Fields  of  Characteristic Two
        if a == 0:  # better way of checking 0...?
            raise ValueError("inverting zero")
        if self.subfield is None:
            return a
        if self.subfield._is_valid(a):
            return self.subfield.inverse(a)
        a0, a1 = self.to_subfield_tuple(a)
        delta = self.subfield.multiply(a0, a0 ^ a1) ^ self.subfield._multiply_alpha(self.subfield.square(a1))
        delta_inv = self.subfield.inverse(delta)
        inv0 = self.subfield.multiply(delta_inv, a0 ^ a1)
        inv1 = self.subfield.multiply(delta_inv, a1)
        return self.from_subfield_tuple((inv0, inv1))


def fanpaar_multiply_recursive(field: BinaryTowerField, subfield: BinaryTowerField, a: int, b: int) -> int:
    # recursive tower mult; uses 2×2 Karatsuba at each step
    # https://en.wikipedia.org/wiki/Karatsuba_algorithm
    a0, a1 = field.to_subfield_tuple(a)
    b0, b1 = field.to_subfield_tuple(b)
    z0 = subfield.multiply(a0, b0)
    z2 = subfield.multiply(a1, b1)
    z1 = subfield.multiply(a0 ^ a1, b0 ^ b1) ^ z0 ^ z2
    z2a = subfield._multiply_alpha(z2)  # this mult is by a constant, unlike the others
    return field.from_subfield_tuple((z0 ^ z2, z1 ^ z2a))


def fanpaar_multiply_alpha_recursive(field: BinaryTowerField, subfield: BinaryTowerField, a: int) -> int:
    a0, a1 = field.to_subfield_tuple(a)
    z1 = subfield._multiply_alpha(a1)
    return field.from_subfield_tuple((a1, a0 ^ z1))


def fanpaar_square_recursive(field: BinaryTowerField, subfield: BinaryTowerField, a: int) -> int:
    a0, a1 = field.to_subfield_tuple(a)
    z0 = subfield.square(a0)
    z2 = subfield.square(a1)
    z2a = subfield._multiply_alpha(z2)
    return field.from_subfield_tuple((z0 ^ z2, z2a))


def fanpaar_inverse_recursive(field: BinaryTowerField, subfield: BinaryTowerField, a: int):
    if subfield._is_valid(a):
        return subfield.inverse(a)
    a0, a1 = field.to_subfield_tuple(a)
    intermediate = a0 ^ subfield._multiply_alpha(a1)
    delta = subfield.multiply(a0, intermediate) ^ subfield.square(a1)
    delta_inv = subfield.inverse(delta)
    inv0 = subfield.multiply(delta_inv, intermediate)
    inv1 = subfield.multiply(delta_inv, a1)
    return field.from_subfield_tuple((inv0, inv1))


class FanPaarTowerField(BinaryTowerField):
    """
    The polynomial modulus for step n of the tower is Xₙ² + Xₙ₋₁ ⋅ Xₙ + 1.
    """

    subfield: FanPaarTowerField | None
    generators = {  # class member
        1: 0x1,
        2: 0x2,
        4: 0xB,
        8: 0x2D,
        16: 0xE2DE,
        32: 0x03E21CEA,
        64: 0x070F870DCD9C1D88,
        128: 0x2E895399AF449ACE499596F6E5FCCAFA,
    }

    def __init__(self, height: int) -> None:
        super().__init__(1 << height)
        self.fmt = f"FanPaarTowerField({self.fmt})"
        if height == 0:
            self.subfield = None
        else:
            self.subfield = FanPaarTowerField(height - 1)

    def multiplicative_generator(self) -> int:
        """Returns a generator of the multiplicative group of units.

        This method is very fast, essentially a lookup of precomputed constants.

        :raises NotImplementedError: if the multiplicative generator for this field is not precomputed
        """
        if self.degree > 128:
            raise NotImplementedError
        return self.generators[self.degree]

    def multiply(self, a: int, b: int) -> int:
        if self.subfield is None:  # base case
            return a & b  # single-bit AND gate
        return fanpaar_multiply_recursive(self, self.subfield, a, b)

    def _multiply_alpha(self, a: int) -> int:
        if self.subfield is None:  # base case
            return a
        return fanpaar_multiply_alpha_recursive(self, self.subfield, a)

    def square(self, a: int) -> int:
        if self.subfield is None:  # base case
            return a
        return fanpaar_square_recursive(self, self.subfield, a)

    def inverse(self, a: int) -> int:
        # Fan and Paar. On Efficient  Inversion in  Tower Fields  of  Characteristic Two
        if a == 0:  # better way of checking 0...?
            raise ValueError("inverting zero")
        if self.subfield is None:
            return a
        return fanpaar_inverse_recursive(self, self.subfield, a)


class Tower192Field(BinaryTowerField):
    """
    extension of 64-bit FanPaar field, by the polynomial X^3 + X + 1.
    uses one level of Toom-3, plus recursive mults for the sub-mults.
    """

    subfield: FanPaarTowerField

    def __init__(self):
        super().__init__(192)
        self.fmt = f"Tower192Field({self.fmt})"
        self.subfield = FanPaarTowerField(6)

    def multiply(self, a: int, b: int) -> int:
        # toom-3 https://en.wikipedia.org/wiki/Toom%E2%80%93Cook_multiplication
        a0, a1, a2 = self.to_subfield_tuple(a)
        b0, b1, b2 = self.to_subfield_tuple(b)
        p0 = a0
        p1 = a0 ^ a1 ^ a2
        p2 = a0 ^ self.subfield.multiply(2, a1) ^ self.subfield.multiply(3, a2)  # TODO: these are big * small mults
        p3 = a0 ^ self.subfield.multiply(3, a1) ^ self.subfield.multiply(2, a2)  # can be done more efficiently
        pinf = a2
        q0 = b0
        q1 = b0 ^ b1 ^ b2
        q2 = b0 ^ self.subfield.multiply(2, b1) ^ self.subfield.multiply(3, b2)  # TODO: these are big * small mults
        q3 = b0 ^ self.subfield.multiply(3, b1) ^ self.subfield.multiply(2, b2)  # can be done more efficiently
        qinf = b2
        # begin recursive mults
        r0 = self.subfield.multiply(p0, q0)
        r1 = self.subfield.multiply(p1, q1)
        r2 = self.subfield.multiply(p2, q2)
        r3 = self.subfield.multiply(p3, q3)
        rinf = self.subfield.multiply(pinf, qinf)
        # begin reconstruction
        # here is the matrix we're going to use to go from evaluations --> coefficients of the pointwise product.
        # sage: m
        # [        1         0         0         0         0]
        # [        1         1         1         1         1]
        # [        1     x0bar x0bar + 1         1     x0bar]
        # [        1 x0bar + 1     x0bar         1 x0bar + 1]
        # [        0         0         0         0         1]
        # sage: m.inverse()
        # [        1         0         0         0         0]
        # [        0         1 x0bar + 1     x0bar         1]
        # [        0         1     x0bar x0bar + 1         0]
        # [        1         1         1         1         0]
        # [        0         0         0         0         1]
        # s :== lower_matrix * r.
        s0 = r0
        s1 = r1 ^ self.subfield.multiply(3, r2) ^ self.subfield.multiply(2, r3) ^ rinf  # these are big * small mults
        s2 = r1 ^ self.subfield.multiply(2, r2) ^ self.subfield.multiply(3, r3)  # can be done more efficiently
        s3 = r0 ^ r1 ^ r2 ^ r3
        s4 = rinf
        # begin reduction by modulus. since Y^3 + Y + 1 has a "gap" (missing the Y^2 term),
        # we can knock out both high monomials, Y^4 and Y^3, in one shot.
        # that is, we need to subtract off s_4 * Y * (Y^3 + Y + 1) + s_3 * (Y^3 + Y + 1).
        # this amounts to s2 -= s4, s1 -= s4 + s3, s0 -= s3
        return self.from_subfield_tuple((s0 ^ s3, s1 ^ s3 ^ s4, s2 ^ s4))

    def square(self, a: int) -> int:
        a0, a1, a2 = self.to_subfield_tuple(a)
        s0 = self.subfield.square(a0)
        s2 = self.subfield.square(a1)
        s4 = self.subfield.square(a2)
        return self.from_subfield_tuple((s0, s4, s2 ^ s4))

    def inverse(self, a: int) -> int:
        # cost (all ops over 64-bit subfield): 3 squarings, 9 mults, plus 1 inversion
        # seems to be just around double the cost of a single multiplication, roughly.
        a0, a1, a2 = self.to_subfield_tuple(a)
        intermediate0 = self.subfield.multiply(a1, a2)
        intermediate1 = self.subfield.square(a1)
        m0 = self.subfield.square(a0 ^ a2) ^ intermediate0 ^ intermediate1
        m1 = self.subfield.multiply(a1, a0 ^ a2) ^ intermediate0 ^ self.subfield.square(a2)
        m2 = intermediate1 ^ self.subfield.multiply(a0 ^ a2, a2)
        delta = self.subfield.multiply(a0, m0) ^ self.subfield.multiply(a2, m1) ^ self.subfield.multiply(a1, m2)
        delta_inv = self.subfield.inverse(delta)
        inv0 = self.subfield.multiply(delta_inv, m0)
        inv1 = self.subfield.multiply(delta_inv, m1)
        inv2 = self.subfield.multiply(delta_inv, m2)
        return self.from_subfield_tuple((inv0, inv1, inv2))


class AESTowerField(BinaryTowerField):
    """
    AESTowerField where the lowest field is F_256 = F_2[X]/(X^8 + X^4 + X^3 + X + 1) with alpha being the map of 0x10
    from FanPaarTowerField8b into AES Field and each subsequent tower is created by the modulus X_n^2 + X_n*a + 1
    where a is 0x100 for 16b 0x10000 for 32b and so on.
    """

    subfield: AESTowerField | None
    generators = {
        8: 0xD0,
        16: 0x4745,
        32: 0xBD478FAB,
        64: 0x0DE1555D2BD78EB4,
        128: 0x6DB54066349EDB96C33A87244A742678,
    }
    gf_aes = GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
    aes_1 = gf_aes(1)

    def __init__(self, height: int) -> None:
        assert height > 2
        super().__init__(1 << height)
        self.fmt = f"AESTowerField({self.fmt})"
        if height == 3:
            self.alpha = 0xD3
            self.subfield = None
        else:
            self.alpha = 1 << (self.degree // 2)
            self.subfield = AESTowerField(height - 1)

    def multiplicative_generator(self) -> int:
        if self.degree > 128:
            raise NotImplementedError
        return self.generators[self.degree]

    def multiply(self, a: int, b: int) -> int:
        if self.subfield is None:
            # Do Normal aes multiplication here
            return int(self.gf_aes(a) * self.gf_aes(b))
        return fanpaar_multiply_recursive(self, self.subfield, a, b)

    def _multiply_alpha(self, a: int) -> int:
        if self.subfield is None:
            return self.multiply(self.alpha, a)
        return fanpaar_multiply_alpha_recursive(self, self.subfield, a)

    def square(self, a: int) -> int:
        if self.subfield is None:  # base case
            # Do normal aes squaring here
            a_ = self.gf_aes(a)
            return int(a_ * a_)
        return fanpaar_square_recursive(self, self.subfield, a)

    def inverse(self, a: int) -> int:
        if a == 0:
            raise ValueError("inverting zero")
        if self.subfield is None:
            # Do normal aes inversion here.
            return int(self.aes_1 / self.gf_aes(a))
        return fanpaar_inverse_recursive(self, self.subfield, a)


class BinaryTowerFieldElem(FiniteFieldElem[int]):
    field: ClassVar[BinaryTowerField]

    def to_subfield_tuple(self) -> tuple[Self, ...]:
        val0, val1 = self.field.to_subfield_tuple(self.value)
        return self.__class__(val0), self.__class__(val1)

    @classmethod
    def from_subfield_tuple(cls, a: tuple[Self, ...]) -> Self:
        return cls(cls.field.from_subfield_tuple(tuple(elem.value for elem in a)))

    def downcast(self, subfield_ty: type[F]) -> F:
        if self.field.degree < subfield_ty.field.degree:
            raise ValueError("cannot downcast to a bigger field")
        if not subfield_ty.field._is_valid(self.value):
            raise ValueError("not a subfield element")
        return subfield_ty(self.value)

    def upcast(self, extfield_ty: type[F]) -> F:
        if extfield_ty.field.degree < self.field.degree:
            raise ValueError("cannot upcast to a smaller field")
        return extfield_ty(self.value)
