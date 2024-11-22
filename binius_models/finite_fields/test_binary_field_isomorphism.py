# (C) 2024 Irreducible Inc.

import numpy as np
from galois import FieldArray
from hypothesis import given
from hypothesis import strategies as st

from .binary_field_isomorphism import (
    GF2_128_POLYVAL,
    POLYVAL_TO_MONTGOMERY,
    gf2_8_fan_paar_to_monomial_isomorphism,
    gf2_8_fast_to_monomial_isomorphism,
    gf2_8_monomial_to_fan_paar_isomorphism,
    gf2_8_monomial_to_fast_isomorphism,
    gf2_128_fan_paar_to_monomial_isomorphism,
    gf2_128_fan_paar_to_montgomery_polyval_isomorphism,
    gf2_128_fan_paar_to_polyval_isomorphism,
    gf2_128_monomial_to_fan_paar_isomorphism,
    gf2_128_montgomery_polyval_to_fan_paar_isomorphism,
    gf2_128_polyval_to_fan_paar_isomorphism,
)
from .tower import BinaryTowerFieldElem, FanPaarTowerField, FASTowerField


class Elem8bFAST(BinaryTowerFieldElem):
    field = FASTowerField(3)


class Elem8bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(3)


class Elem128bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


@given(
    a_val=st.integers(0, 2**8 - 1),
    b_val=st.integers(0, 2**8 - 1),
)
def test_gf2_8_fast_isomorphism(a_val: int, b_val: int):
    a = Elem8bFAST(a_val)
    b = Elem8bFAST(b_val)
    a_mono = gf2_8_fast_to_monomial_isomorphism(a_val)
    b_mono = gf2_8_fast_to_monomial_isomorphism(b_val)
    ab_mono = a_mono * b_mono
    assert isinstance(ab_mono, FieldArray)
    assert (a * b).value == gf2_8_monomial_to_fast_isomorphism(ab_mono)


@given(
    a_val=st.integers(0, 2**8 - 1),
    b_val=st.integers(0, 2**8 - 1),
)
def test_gf2_8_fan_paar_isomorphism(a_val: int, b_val: int):
    a = Elem8bFP(a_val)
    b = Elem8bFP(b_val)
    a_mono = gf2_8_fan_paar_to_monomial_isomorphism(a_val)
    b_mono = gf2_8_fan_paar_to_monomial_isomorphism(b_val)
    ab_mono = a_mono * b_mono
    assert isinstance(ab_mono, FieldArray)
    assert (a * b).value == gf2_8_monomial_to_fan_paar_isomorphism(ab_mono)


@given(
    a_val=st.integers(0, 2**128 - 1),
    b_val=st.integers(0, 2**128 - 1),
)
def test_gf2_128_fan_paar_monomial_isomorphism(a_val: int, b_val: int):
    a = Elem128bFP(a_val)
    b = Elem128bFP(b_val)
    a_mono = gf2_128_fan_paar_to_monomial_isomorphism(a_val)
    b_mono = gf2_128_fan_paar_to_monomial_isomorphism(b_val)
    ab_mono = a_mono * b_mono
    assert isinstance(ab_mono, FieldArray)
    assert (a * b).value == gf2_128_monomial_to_fan_paar_isomorphism(ab_mono)


@given(
    a_val=st.integers(0, 2**128 - 1),
    b_val=st.integers(0, 2**128 - 1),
)
def test_gf2_128_fan_paar_polyval_isomorphism(a_val: int, b_val: int):
    a = Elem128bFP(a_val)
    b = Elem128bFP(b_val)
    a_polyval = gf2_128_fan_paar_to_polyval_isomorphism(a_val)
    b_polyval = gf2_128_fan_paar_to_polyval_isomorphism(b_val)
    ab_polyval = a_polyval * b_polyval
    assert isinstance(ab_polyval, FieldArray)
    assert (a * b).value == gf2_128_polyval_to_fan_paar_isomorphism(ab_polyval)


@given(a_val=st.integers(0, 2**128 - 1))
def test_gf2_128_polyval_to_montgomery(a_val: int):
    a = GF2_128_POLYVAL(a_val)
    expected = a * GF2_128_POLYVAL("x^127 + x^126 + x^121 + 1")
    assert GF2_128_POLYVAL.Vector(POLYVAL_TO_MONTGOMERY @ a.vector()) == expected


@given(
    a_val=st.integers(0, 2**128 - 1),
    b_val=st.integers(0, 2**128 - 1),
)
def test_gf2_128_fan_paar_montgomery_polyval_isomorphism(a_val: int, b_val: int):
    a = Elem128bFP(a_val)
    b = Elem128bFP(b_val)
    a_polyval = gf2_128_fan_paar_to_montgomery_polyval_isomorphism(a_val)
    b_polyval = gf2_128_fan_paar_to_montgomery_polyval_isomorphism(b_val)
    r_inv = np.reciprocal(GF2_128_POLYVAL("x^127 + x^126 + x^121 + 1"))
    assert gf2_128_montgomery_polyval_to_fan_paar_isomorphism(a_polyval * b_polyval * r_inv) == (a * b).value
