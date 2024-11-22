from typing import cast

import numpy as np
from galois import GF, GF2, FieldArray

from ..utils.utils import int_to_bits
from .tower import BinaryTowerFieldElem, FanPaarTowerField, FASTowerField


class Elem8bFAST(BinaryTowerFieldElem):
    field = FASTowerField(3)


class Elem8bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(3)


class Elem128bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


GF2_8 = GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
GF2_128 = GF(2**128, irreducible_poly="x^128 + x^7 + x^2 + x + 1")
GF2_128_POLYVAL = GF(2**128, irreducible_poly="x^128 + x^127 + x^126 + x^121 + 1")


def root_to_matrix(root: BinaryTowerFieldElem) -> FieldArray:
    field = root.field
    n_bits = field.degree
    accum = type(root).one()
    entries = []
    for _ in range(n_bits):
        entries.append(int_to_bits(accum.value, n_bits))
        accum *= root
    return cast(FieldArray, np.transpose(GF2(entries)))


GF2_8_MONOMIAL_TO_FAST_ROOT = Elem8bFAST(0x56)  # hardcode: root of monomial in tower
GF2_8_MONOMIAL_TO_FAST_ISOMORPHISM = np.flip(root_to_matrix(GF2_8_MONOMIAL_TO_FAST_ROOT), axis=(0, 1))
GF2_8_FAST_TO_MONOMIAL_ISOMORPHISM = np.linalg.inv(GF2_8_MONOMIAL_TO_FAST_ISOMORPHISM)


def gf2_8_fast_to_monomial_isomorphism(value: int) -> FieldArray:
    v = GF2_8(value).vector()
    return GF2_8.Vector(GF2_8_FAST_TO_MONOMIAL_ISOMORPHISM @ v)


def gf2_8_monomial_to_fast_isomorphism(value: FieldArray) -> int:
    return int(GF2_8.Vector(GF2_8_MONOMIAL_TO_FAST_ISOMORPHISM @ value.vector()))


GF2_8_MONOMIAL_TO_FAN_PAAR_ROOT = Elem8bFP(0x3C)  # hardcode: root of monomial in tower
GF2_8_MONOMIAL_TO_FAN_PAAR_ISOMORPHISM = np.flip(root_to_matrix(GF2_8_MONOMIAL_TO_FAN_PAAR_ROOT), axis=(0, 1))
GF2_8_FAN_PAAR_TO_MONOMIAL_ISOMORPHISM = np.linalg.inv(GF2_8_MONOMIAL_TO_FAN_PAAR_ISOMORPHISM)


def gf2_8_fan_paar_to_monomial_isomorphism(value: int) -> FieldArray:
    v = GF2_8(value).vector()
    return GF2_8.Vector(GF2_8_FAN_PAAR_TO_MONOMIAL_ISOMORPHISM @ v)


def gf2_8_monomial_to_fan_paar_isomorphism(value: FieldArray) -> int:
    return int(GF2_8.Vector(GF2_8_MONOMIAL_TO_FAN_PAAR_ISOMORPHISM @ value.vector()))


def gf2_8_tower_to_fan_paar_tower_isomorphism(value: int) -> int:
    num_bits = value.bit_length()
    num_bytes = (num_bits + 7) // 8
    final_value = 0
    for i in range(num_bytes):
        lim = value & 0xFF
        value >>= 8
        final_value |= gf2_8_monomial_to_fan_paar_isomorphism(GF2_8(lim)) << 8 * i
    return final_value


GF2_128_MONOMIAL_TO_FAN_PAAR_ROOT = Elem128bFP(0x6EBDCF735FAD51FA97E40CF3F6B068D4)
GF2_128_MONOMIAL_TO_FAN_PAAR_ISOMORPHISM = np.flip(root_to_matrix(GF2_128_MONOMIAL_TO_FAN_PAAR_ROOT), axis=(0, 1))
GF2_128_FAN_PAAR_TO_MONOMIAL_ISOMORPHISM = np.linalg.inv(GF2_128_MONOMIAL_TO_FAN_PAAR_ISOMORPHISM)


GF2_128_POLYVAL_TO_FAN_PAAR_ROOT = Elem128bFP(0xACC045053A949202B5B1AFA15441D009)
GF2_128_POLYVAL_TO_FAN_PAAR_ISOMORPHISM = np.flip(root_to_matrix(GF2_128_POLYVAL_TO_FAN_PAAR_ROOT), axis=(0, 1))
GF2_128_FAN_PAAR_TO_POLYVAL_ISOMORPHISM = np.linalg.inv(GF2_128_POLYVAL_TO_FAN_PAAR_ISOMORPHISM)


MONTGOMERY_POLYVAL_R = GF2_128_POLYVAL("x^127 + x^126 + x^121 + 1")

POLYVAL_TO_MONTGOMERY = np.transpose(
    GF2([cast(FieldArray, GF2_128_POLYVAL(1 << i) * MONTGOMERY_POLYVAL_R).vector() for i in reversed(range(128))])
)

GF2_128_FAN_PAAR_TO_MONTGOMERY_POLYVAL_ISOMORPHISM = POLYVAL_TO_MONTGOMERY @ GF2_128_FAN_PAAR_TO_POLYVAL_ISOMORPHISM

GF2_128_MONTGOMERY_POLYVAL_TO_FAN_PAAR_ISOMORPHISM = np.linalg.inv(GF2_128_FAN_PAAR_TO_MONTGOMERY_POLYVAL_ISOMORPHISM)


def gf2_128_fan_paar_to_monomial_isomorphism(value: int) -> FieldArray:
    v = GF2_128(value).vector()
    return GF2_128.Vector(GF2_128_FAN_PAAR_TO_MONOMIAL_ISOMORPHISM @ v)


def gf2_128_monomial_to_fan_paar_isomorphism(value: FieldArray) -> int:
    return int(GF2_128.Vector(GF2_128_MONOMIAL_TO_FAN_PAAR_ISOMORPHISM @ value.vector()))


def gf2_128_fan_paar_to_polyval_isomorphism(value: int) -> FieldArray:
    v = GF2_128_POLYVAL(value).vector()
    return GF2_128_POLYVAL.Vector(GF2_128_FAN_PAAR_TO_POLYVAL_ISOMORPHISM @ v)


def gf2_128_polyval_to_fan_paar_isomorphism(value: FieldArray) -> int:
    return int(GF2_128_POLYVAL.Vector(GF2_128_POLYVAL_TO_FAN_PAAR_ISOMORPHISM @ value.vector()))


def gf2_128_fan_paar_to_montgomery_polyval_isomorphism(value: int) -> FieldArray:
    v = GF2_128_POLYVAL(value).vector()
    return GF2_128_POLYVAL.Vector(GF2_128_FAN_PAAR_TO_MONTGOMERY_POLYVAL_ISOMORPHISM @ v)


def gf2_128_montgomery_polyval_to_fan_paar_isomorphism(value: FieldArray) -> int:
    return int(GF2_128_POLYVAL.Vector(GF2_128_MONTGOMERY_POLYVAL_TO_FAN_PAAR_ISOMORPHISM @ value.vector()))
