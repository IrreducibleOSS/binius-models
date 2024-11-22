# (C) 2024 Irreducible Inc.

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .tower import BinaryTowerFieldElem, FanPaarTowerField, FASTowerField
from .tower_packed import (
    _fast_tower_field_multiply_alpha_simd,
    _flip,
    _generate_interleave_mask,
    _multiply_alpha_simd,
    fast_tower_field_inverse_simd,
    fast_tower_field_multiply_simd,
    fast_tower_field_square_simd,
    interleave,
    inverse_simd,
    multiply_simd,
    square_simd,
)


class Elem1bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(0)


class Elem4bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(2)


class Elem32bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(5)


class Elem1bFAST(BinaryTowerFieldElem):
    field = FASTowerField(0)


class Elem4bFAST(BinaryTowerFieldElem):
    field = FASTowerField(2)


class Elem32bFAST(BinaryTowerFieldElem):
    field = FASTowerField(5)


def test_generate_interleave_mask():
    assert _generate_interleave_mask(0, 7) == 0x55555555555555555555555555555555
    assert _generate_interleave_mask(3, 4) == 0x00FF00FF00FF00FF00FF00FF00FF00FF
    assert _generate_interleave_mask(6, 1) == 0x0000000000000000FFFFFFFFFFFFFFFF


def test_interleave():
    a = 0x0000000000000000FFFFFFFFFFFFFFFF
    b = 0xFFFFFFFFFFFFFFFF0000000000000000

    c = 0xAAAAAAAAAAAAAAAA5555555555555555
    d = 0xAAAAAAAAAAAAAAAA5555555555555555
    assert interleave(0, 7, a, b) == (c, d)
    assert interleave(0, 7, c, d) == (a, b)

    c = 0xCCCCCCCCCCCCCCCC3333333333333333
    d = 0xCCCCCCCCCCCCCCCC3333333333333333
    assert interleave(1, 6, a, b) == (c, d)
    assert interleave(1, 6, c, d) == (a, b)

    c = 0xF0F0F0F0F0F0F0F00F0F0F0F0F0F0F0F
    d = 0xF0F0F0F0F0F0F0F00F0F0F0F0F0F0F0F
    assert interleave(2, 5, a, b) == (c, d)
    assert interleave(2, 5, c, d) == (a, b)

    a = 0x0F0E0D0C0B0A09080706050403020100
    b = 0x1F1E1D1C1B1A19181716151413121110

    c = 0x1E0E1C0C1A0A18081606140412021000
    d = 0x1F0F1D0D1B0B19091707150513031101
    assert interleave(3, 4, a, b) == (c, d)
    assert interleave(3, 4, c, d) == (a, b)

    c = 0x1D1C0D0C191809081514050411100100
    d = 0x1F1E0F0E1B1A0B0A1716070613120302
    assert interleave(4, 3, a, b) == (c, d)
    assert interleave(4, 3, c, d) == (a, b)

    c = 0x1B1A19180B0A09081312111003020100
    d = 0x1F1E1D1C0F0E0D0C1716151407060504
    assert interleave(5, 2, a, b) == (c, d)
    assert interleave(5, 2, c, d) == (a, b)

    c = 0x17161514131211100706050403020100
    d = 0x1F1E1D1C1B1A19180F0E0D0C0B0A0908
    assert interleave(6, 1, a, b) == (c, d)
    assert interleave(6, 1, c, d) == (a, b)


def test_flip():
    a = 0xFEDCBA9876543210

    b = 0xFDEC7564B9A83120
    assert _flip(0, 6, a) == b
    assert _flip(0, 6, b) == a

    b = 0xFB73EA62D951C840
    assert _flip(1, 5, a) == b
    assert _flip(1, 5, b) == a

    b = 0xEFCDAB8967452301
    assert _flip(2, 4, a) == b
    assert _flip(2, 4, b) == a

    b = 0xDCFE98BA54761032
    assert _flip(3, 3, a) == b
    assert _flip(3, 3, b) == a

    b = 0xBA98FEDC32107654
    assert _flip(4, 2, a) == b
    assert _flip(4, 2, b) == a

    b = 0x76543210FEDCBA98
    assert _flip(5, 1, a) == b
    assert _flip(5, 1, b) == a


@given(
    a_val=st.integers(0, 2**128 - 1),
    b_val=st.integers(0, 2**128 - 1),
)
@settings(deadline=10000)
@pytest.mark.parametrize("elem", [Elem1bFP, Elem4bFP, Elem32bFP])
def test_multiply_simd(elem: type[BinaryTowerFieldElem], a_val: int, b_val: int):
    height = int(math.log2(elem.field.degree))
    log_width = 7 - height

    c_val = multiply_simd(height, log_width, a_val, b_val)

    mask = 2 ** (2**height) - 1
    for i in range(2**log_width):
        a_elem = Elem32bFP((a_val >> (2**height * i)) & mask)
        b_elem = Elem32bFP((b_val >> (2**height * i)) & mask)
        c_elem = Elem32bFP((c_val >> (2**height * i)) & mask)
        assert a_elem * b_elem == c_elem


@given(
    a_val=st.integers(0, 2**128 - 1),
)
@settings(deadline=10000)
@pytest.mark.parametrize("elem", [Elem1bFP, Elem4bFP, Elem32bFP])
def test__multiply_alpha_simd(elem: type[BinaryTowerFieldElem], a_val: int):
    height = int(math.log2(elem.field.degree))
    log_width = 7 - height

    res_val = _multiply_alpha_simd(height, log_width, a_val)

    mask = 2 ** (2**height) - 1
    for i in range(2**log_width):
        a_elem = int((a_val >> (2**height * i)) & mask)
        res_elem = int((res_val >> (2**height * i)) & mask)
        assert res_elem == elem.field._multiply_alpha(a_elem)


@given(
    a_val=st.integers(0, 2**128 - 1),
)
@settings(deadline=10000)
@pytest.mark.parametrize("elem", [Elem1bFP, Elem4bFP, Elem32bFP])
def test_square_simd(elem: type[BinaryTowerFieldElem], a_val: int):
    height = int(math.log2(elem.field.degree))
    log_width = 7 - height

    res_val = square_simd(height, log_width, a_val)

    mask = 2 ** (2**height) - 1
    for i in range(2**log_width):
        a_elem = Elem32bFP((a_val >> (2**height * i)) & mask)
        res_elem = Elem32bFP((res_val >> (2**height * i)) & mask)
        assert res_elem == a_elem * a_elem


@given(
    a_val=st.integers(0, 2**128 - 1),
)
@settings(deadline=10000)
@pytest.mark.parametrize("elem", [Elem1bFP, Elem4bFP, Elem32bFP])
def test_inverse_simd(elem: type[BinaryTowerFieldElem], a_val: int):
    height = int(math.log2(elem.field.degree))
    log_width = 7 - height

    res_val = inverse_simd(height, log_width, a_val)

    mask = 2 ** (2**height) - 1
    for i in range(2**log_width):
        a_elem = Elem32bFP((a_val >> (2**height * i)) & mask)
        res_elem = Elem32bFP((res_val >> (2**height * i)) & mask)
        assert (a_elem.is_zero() and res_elem.is_zero()) or (res_elem * a_elem == elem.one())


@given(
    a_val=st.integers(0, 2**128 - 1),
)
@settings(deadline=10000)
@pytest.mark.parametrize("elem", [Elem1bFAST, Elem4bFAST, Elem32bFAST])
def test__fast_tower_field_multiply_alpha_simd(elem: type[BinaryTowerFieldElem], a_val: int):
    height = int(math.log2(elem.field.degree))
    log_width = 7 - height

    res_val = _fast_tower_field_multiply_alpha_simd(height, log_width, a_val)

    mask = 2 ** (2**height) - 1
    for i in range(2**log_width):
        a_elem = int((a_val >> (2**height * i)) & mask)
        res_elem = int((res_val >> (2**height * i)) & mask)
        assert bin(res_elem) == bin(elem.field._multiply_alpha(a_elem))


a = 15


@given(
    a_val=st.integers(0, 2**128 - 1),
    b_val=st.integers(0, 2**128 - 1),
)
@settings(deadline=10000)
@pytest.mark.parametrize("elem", [Elem1bFAST, Elem4bFAST, Elem32bFAST])
def test_fast_tower_field_multiply_simd(elem: type[BinaryTowerFieldElem], a_val: int, b_val: int):
    height = int(math.log2(elem.field.degree))
    log_width = 7 - height

    c_val = fast_tower_field_multiply_simd(height, log_width, a_val, b_val)

    mask = 2 ** (2**height) - 1
    for i in range(2**log_width):
        a_elem = Elem32bFAST((a_val >> (2**height * i)) & mask)
        b_elem = Elem32bFAST((b_val >> (2**height * i)) & mask)
        c_elem = Elem32bFAST((c_val >> (2**height * i)) & mask)
        assert a_elem * b_elem == c_elem


@given(
    a_val=st.integers(0, 2**128 - 1),
)
@settings(deadline=10000)
@pytest.mark.parametrize("elem", [Elem1bFAST, Elem4bFAST, Elem32bFAST])
def test_fast_tower_field_square_simd(elem: type[BinaryTowerFieldElem], a_val: int):
    height = int(math.log2(elem.field.degree))
    log_width = 7 - height

    res_val = fast_tower_field_square_simd(height, log_width, a_val)

    mask = 2 ** (2**height) - 1
    for i in range(2**log_width):
        a_elem = Elem32bFAST((a_val >> (2**height * i)) & mask)
        res_elem = Elem32bFAST((res_val >> (2**height * i)) & mask)
        assert res_elem == a_elem * a_elem


@given(
    a_val=st.integers(0, 2**128 - 1),
)
@settings(deadline=10000)
@pytest.mark.parametrize("elem", [Elem1bFAST, Elem4bFAST, Elem32bFAST])
def test_fast_tower_field_inverse_simd(elem: type[BinaryTowerFieldElem], a_val: int):
    height = int(math.log2(elem.field.degree))
    log_width = 7 - height

    res_val = fast_tower_field_inverse_simd(height, log_width, a_val)

    mask = 2 ** (2**height) - 1
    for i in range(2**log_width):
        a_elem = Elem32bFAST((a_val >> (2**height * i)) & mask)
        res_elem = Elem32bFAST((res_val >> (2**height * i)) & mask)
        assert (a_elem.is_zero() and res_elem.is_zero()) or (res_elem * a_elem == elem.one())
