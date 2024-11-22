from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ..finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField
from ..utils.utils import bits_mask, int_to_bits
from .shift import ShiftIndicator, evaluate_shift_polynomial


class Elem1b(BinaryTowerFieldElem):
    field = FanPaarTowerField(0)


class Elem128b(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


@given(x=st.integers(0, 2**8 - 1), o=st.integers(0, 2**8 - 1))
@settings(deadline=None)
def test_naive_indicator_true(x: int, o: int) -> None:
    v = 8
    b = 8
    indicator = ShiftIndicator(Elem1b, v, b, o)
    y = x + o & bits_mask(8)
    x_bits = [Elem1b(b) for b in int_to_bits(x, 8)]
    y_bits = [Elem1b(b) for b in int_to_bits(y, 8)]
    assert indicator.evaluate_at_point(x_bits, y_bits) == Elem1b.one()


@given(x=st.integers(0, 2**8 - 1), y=st.integers(0, 2**8 - 1), o=st.integers(0, 2**8 - 1))
@settings(deadline=None)
def test_naive_indicator_false(x: int, y: int, o: int) -> None:
    v = 8
    b = 8
    indicator = ShiftIndicator(Elem1b, v, b, o)
    assume(y != x + o & bits_mask(8))
    x_bits = [Elem1b(b) for b in int_to_bits(x, 8)]
    y_bits = [Elem1b(b) for b in int_to_bits(y, 8)]
    assert indicator.evaluate_at_point(x_bits, y_bits) == Elem1b.zero()


def test_indicator_correctness() -> None:
    v = 5  # length 32
    b = 3  # 8-element blocks
    o = 6
    indicator = ShiftIndicator(Elem128b, v, b, o)

    r = [Elem128b.random() for _ in range(v)]
    array = indicator.evaluate_over_hypercube(r)

    for i in range(1 << b):
        x = [Elem128b(value) for value in int_to_bits(i, v)]
        assert array[i] == indicator.evaluate_at_point(x, r)


def test_shift_polynomial_correctness() -> None:
    # in the special case that we evaluate shift(f)(r) at a point r IN the cube,
    # the result should be f[u] for an appropriate u (i.e., some shift of r).
    v = 5  # length 32
    b = 3  # 8-element blocks
    o = 6
    indicator = ShiftIndicator(Elem128b, v, b, o)

    f = [Elem128b.random() for _ in range(1 << v)]

    for i in range(1 << v):
        r = [Elem128b(value) for value in int_to_bits(i, v)]
        value = evaluate_shift_polynomial(indicator, f, r)

        u = (i & (-1 << b)) | ((i - o) & (1 << b) - 1)  # highest v - b bits are equal to i. lowest are equal to i - o.
        assert value == f[u]
