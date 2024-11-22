# (C) 2024 Irreducible Inc.

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from binius_models.tests.helpers import random_integers_strategy

from .tower import BinaryTowerFieldElem, FanPaarTowerField, Tower192Field


class Elem192b(BinaryTowerFieldElem):
    field = Tower192Field()


class Elem64bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(6)


class Elem32bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(5)


@pytest.mark.parametrize_hypothesis(
    slow=(
        settings(deadline=None),
        given(
            a_val=st.integers(0, 2**192 - 1),
            b_val=st.integers(0, 2**192 - 1),
        ),
    ),
    fast=(
        settings(deadline=None, max_examples=1),
        given(
            a_val=random_integers_strategy(1, 2**192 - 1),
            b_val=random_integers_strategy(1, 2**192 - 1),
        ),
    ),
)
def test_commutativity(a_val: int, b_val: int):
    a = Elem192b(a_val)
    b = Elem192b(b_val)
    assert a * b == b * a


@pytest.mark.parametrize_hypothesis(
    slow=(
        settings(deadline=None),
        given(
            a_val=st.integers(0, 2**192 - 1),
            b_val=st.integers(0, 2**192 - 1),
            c_val=st.integers(0, 2**192 - 1),
        ),
    ),
    fast=(
        settings(deadline=None, max_examples=1),
        given(
            a_val=random_integers_strategy(1, 2**192 - 1),
            b_val=random_integers_strategy(1, 2**192 - 1),
            c_val=random_integers_strategy(1, 2**192 - 1),
        ),
    ),
)
def test_associativity(a_val: int, b_val: int, c_val: int):
    a = Elem192b(a_val)
    b = Elem192b(b_val)
    c = Elem192b(c_val)
    assert (a * b) * c == a * (b * c)


@pytest.mark.parametrize_hypothesis(
    slow=(
        settings(deadline=None),
        given(a_val=st.integers(1, 2**192 - 1)),
    ),
    fast=(
        settings(deadline=None, max_examples=1),
        given(a_val=random_integers_strategy(1, 2**192 - 1)),
    ),
)
def test_multiplicative_group(a_val: int):
    a = Elem192b(a_val)
    assert a ** (2**Elem192b.field.degree - 1) == Elem192b.one()  # mult. group has exponent dividing 2ⁿ − 1


@pytest.mark.parametrize_hypothesis(
    slow=(
        settings(deadline=None),
        given(a_val=st.integers(1, 2**192 - 1)),
    ),
    fast=(
        settings(deadline=None, max_examples=1),
        given(a_val=random_integers_strategy(1, 2**192 - 1)),
    ),
)
def test_square_correctness(a_val: int) -> None:
    a = Elem192b(a_val)
    assert a.square() == a * a


@pytest.mark.parametrize_hypothesis(
    slow=(
        settings(deadline=None),
        given(a_val=st.integers(1, 2**192 - 1)),
    ),
    fast=(
        settings(deadline=None, max_examples=1),
        given(a_val=random_integers_strategy(1, 2**192 - 1)),
    ),
)
def test_inversion_correctness(a_val: int) -> None:
    a = Elem192b(a_val)
    assert a * a.inverse() == Elem192b.one()


@pytest.mark.parametrize_hypothesis(
    slow=(
        settings(deadline=None),
        given(a_val=st.integers(1, 2**192 - 1)),
    ),
    fast=(
        settings(deadline=None, max_examples=1),
        given(a_val=random_integers_strategy(1, 2**192 - 1)),
    ),
)
def test_subfields(a_val):
    a = Elem192b(a_val)
    assert a ** (2**Elem192b.field.degree - 1) == Elem192b.one()
    a **= 2**128 + 2**64 + 1  # norm map. 𝔽_{2¹⁹²} → 𝔽_{2⁶⁴}
    a.downcast(Elem64bFP)
    assert a ** (2**Elem64bFP.field.degree - 1) == Elem64bFP.one()
    a **= 2**32 + 1  # norm map 𝔽_{2⁶⁴} → 𝔽_{2³²}
    a.downcast(Elem32bFP)
    assert a ** (2**Elem32bFP.field.degree - 1) == Elem32bFP.one()
