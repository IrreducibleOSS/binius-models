from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ..finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField
from ..utils.utils import int_to_bits
from .equality import EqualityIndicator, evaluate_multilinear_extension


class Elem1b(BinaryTowerFieldElem):
    field = FanPaarTowerField(0)


class Elem128b(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


@given(x=st.integers(0, 2**8 - 1))
@settings(deadline=None)
def test_naive_indicator_true(x: int) -> None:
    v = 8
    indicator = EqualityIndicator(Elem1b, v)
    x_bits = [Elem1b(b) for b in int_to_bits(x, 8)]
    assert indicator.evaluate_at_point(x_bits, x_bits) == Elem1b.one()


@given(x=st.integers(0, 2**8 - 1), y=st.integers(0, 2**8 - 1))
@settings(deadline=None)
def test_naive_indicator_false(x: int, y: int) -> None:
    v = 8
    indicator = EqualityIndicator(Elem1b, v)
    assume(y != x)
    x_bits = [Elem1b(b) for b in int_to_bits(x, 8)]
    y_bits = [Elem1b(b) for b in int_to_bits(y, 8)]
    assert indicator.evaluate_at_point(x_bits, y_bits) == Elem1b.zero()


def test_indicator_correctness() -> None:
    v = 5  # length 32
    indicator = EqualityIndicator(Elem128b, v)

    r = [Elem128b.random() for _ in range(v)]
    array = indicator.evaluate_over_hypercube(r)

    for i in range(1 << v):
        x = [Elem128b(value) for value in int_to_bits(i, v)]
        assert array[i] == indicator.evaluate_at_point(x, r)


def test_mle_correctness() -> None:
    v = 5  # length 32
    indicator = EqualityIndicator(Elem128b, v)

    f = [Elem128b.random() for _ in range(1 << v)]
    r = [Elem128b.random() for _ in range(v)]
    actual = evaluate_multilinear_extension(indicator, f, r)  # efficient, linear alg.

    expected = Elem128b.zero()
    for i in range(1 << v):  # quasilinear, naive alg.
        x = [Elem128b(value) for value in int_to_bits(i, v)]
        expected += f[i] * indicator.evaluate_at_point(x, r)
    assert actual == expected
