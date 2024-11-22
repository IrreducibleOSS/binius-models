# (C) 2024 Irreducible Inc.

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from binius_models.tests.helpers import random_integers_strategy
from binius_models.utils.utils import factorize

from .tower import AESTowerField, BinaryTowerFieldElem, FanPaarTowerField, FASTowerField


class Elem128bFAST(BinaryTowerFieldElem):
    field = FASTowerField(7)


class Elem128bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


class Elem128bAES(BinaryTowerFieldElem):
    field = AESTowerField(7)


class Elem64bFAST(BinaryTowerFieldElem):
    field = FASTowerField(6)


class Elem64bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(6)


class Elem64AES(BinaryTowerFieldElem):
    field = AESTowerField(6)


class Elem32bFAST(BinaryTowerFieldElem):
    field = FASTowerField(5)


class Elem32bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(5)


class Elem32AES(BinaryTowerFieldElem):
    field = AESTowerField(5)


class Elem16bFAST(BinaryTowerFieldElem):
    field = FASTowerField(4)


class Elem16bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(4)


class Elem16AES(BinaryTowerFieldElem):
    field = AESTowerField(4)


@given(
    a_val=st.integers(0, 2**64 - 1),
    b_val=st.integers(0, 2**64 - 1),
)
@pytest.mark.parametrize("elem64b", [Elem64bFP, Elem64bFAST, Elem64AES])
def test_commutativity(elem64b: type[BinaryTowerFieldElem], a_val: int, b_val: int):
    a = elem64b(a_val)
    b = elem64b(b_val)
    assert a * b == b * a


@given(
    a_val=st.integers(0, 2**64 - 1),
    b_val=st.integers(0, 2**64 - 1),
    c_val=st.integers(0, 2**64 - 1),
)
@pytest.mark.parametrize("elem64b", [Elem64bFP, Elem64bFAST, Elem64AES])
def test_associativity(elem64b: type[BinaryTowerFieldElem], a_val: int, b_val: int, c_val: int):
    a = elem64b(a_val)
    b = elem64b(b_val)
    c = elem64b(c_val)
    assert (a * b) * c == a * (b * c)


@pytest.mark.parametrize("elem64b", [Elem64bFP, Elem64bFAST, Elem64AES])
@pytest.mark.parametrize_hypothesis(
    slow=(
        settings(deadline=10000),
        given(a_val=st.integers(1, 2**64 - 1)),
    ),
    fast=(
        settings(deadline=10000, max_examples=1),
        given(a_val=random_integers_strategy(1, 2**64 - 1)),
    ),
)
def test_multiplicative_group(elem64b: type[BinaryTowerFieldElem], a_val: int):
    a = elem64b(a_val)
    assert a ** (2**elem64b.field.degree - 1) == elem64b.one()  # mult. group has exponent dividing 2ⁿ − 1


@given(a_val=st.integers(0, 2**64 - 1))
@pytest.mark.parametrize("elem64b", [Elem64bFP, Elem64bFAST, Elem64AES])
def test_square_correctness(elem64b: type[BinaryTowerFieldElem], a_val: int) -> None:
    a = elem64b(a_val)
    assert a.square() == a * a


@given(a_val=st.integers(1, 2**64 - 1))
@pytest.mark.parametrize("elem64b", [Elem64bFP, Elem64bFAST, Elem64AES])
def test_inversion_correctness(elem64b: type[BinaryTowerFieldElem], a_val: int) -> None:
    a = elem64b(a_val)
    assert a * a.inverse() == elem64b.one()


@pytest.mark.parametrize(
    "elem128b,elem64b,elem32b,elem16b",
    [
        (Elem128bFAST, Elem64bFAST, Elem32bFAST, Elem16bFAST),
        (Elem128bFP, Elem64bFP, Elem32bFP, Elem16bFP),
        (Elem128bAES, Elem64AES, Elem32AES, Elem16AES),
    ],
)
@pytest.mark.parametrize_hypothesis(
    slow=(
        settings(deadline=10000),
        given(a_val=st.integers(1, 2**128 - 1)),
    ),
    fast=(
        settings(deadline=10000, max_examples=1),
        given(a_val=random_integers_strategy(1, 2**128 - 1)),
    ),
)
def test_subfields(
    elem128b: type[BinaryTowerFieldElem],
    elem64b: type[BinaryTowerFieldElem],
    elem32b: type[BinaryTowerFieldElem],
    elem16b: type[BinaryTowerFieldElem],
    a_val: int,
) -> None:
    # tests that the various norm maps properly send elements into the corresponding subfields,
    # and that elements of the subfields have the expected (smaller) multiplicative orders.
    a = elem128b(a_val)
    assert a ** (2**128 - 1) == elem128b.one()
    a **= 2**64 + 1  # norm map 𝔽_{2¹²⁸} → 𝔽_{2⁶⁴}
    a.downcast(elem64b)  # a[0 : tower.n >> 1] == 0
    assert a ** (2**elem64b.field.degree - 1) == elem64b.one()
    a **= 2**32 + 1  # norm map 𝔽_{2⁶⁴} → 𝔽_{2³²}
    a.downcast(elem32b)  # a[0 : tower.n - tower.n >> 2] == 0
    assert a ** (2**elem32b.field.degree - 1) == elem32b.one()
    a **= 2**16 + 1  # norm map 𝔽_{2³²} → 𝔽_{2¹⁶}
    a.downcast(elem16b)  # a[0 : tower.n - tower.n >> 3] == 0
    assert a ** (2**elem16b.field.degree - 1) == elem16b.one()


@pytest.mark.parametrize(
    "elem",
    [
        Elem128bFAST,
        Elem64bFAST,
        Elem32bFAST,
        Elem16bFAST,
        Elem128bFP,
        Elem64bFP,
        Elem32bFP,
        Elem16bFP,
        Elem128bAES,
        Elem64AES,
        Elem32AES,
        Elem16AES,
    ],
)
def test_generator(elem: type[BinaryTowerFieldElem]) -> None:
    a = elem(elem.field.multiplicative_generator())  # how to avoid the cast?
    multiplicative_order = 2**elem.field.degree - 1
    assert a**multiplicative_order == elem.one()
    factors = factorize(multiplicative_order)
    for factor in factors:
        assert a ** (multiplicative_order // factor) != elem.one()
