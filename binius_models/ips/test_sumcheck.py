# (C) 2024 Irreducible Inc.
import pytest

from binius_models.finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField

from .polynomial import Polynomial
from .sumcheck import Sumcheck


class Elem1b(BinaryTowerFieldElem):
    field = FanPaarTowerField(0)


class Elem8b(BinaryTowerFieldElem):
    field = FanPaarTowerField(3)


class Elem128b(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


def run_test(v, d, composition, high_to_low):
    multilinears = [[Elem128b(Elem8b.random().value) for _ in range(1 << v)] for _ in range(d)]
    sumcheck = Sumcheck(Elem128b, multilinears, composition, high_to_low)

    claim = sumcheck.sum()
    for _ in range(sumcheck.v):
        evaluations = sumcheck.compute_round_polynomial()
        evaluations.insert(0, evaluations[0] + claim)
        challenge = Elem128b.random()
        sumcheck.receive_challenge(challenge)
        claim = sumcheck.interpolate(evaluations, challenge)

    assert sumcheck.query() == claim


def test_1_5() -> None:
    # 1 polynomial! of length 2⁵, so over 5 variables.
    v = 5
    d = 1
    composition = Polynomial(Elem128b, d, {tuple([1] * d): Elem128b.one()})  # identity polynomial on 1 variable
    for high_to_low in [True, False]:
        run_test(v, d, composition, high_to_low)


def test_3_5() -> None:
    # 3 polynomials, each of length 2⁵, so over 5 variables.
    v = 5
    d = 3
    composition = Polynomial(Elem128b, d, {tuple([1] * d): Elem128b.one()})  # product monomial on d variables.
    for high_to_low in [True, False]:
        run_test(v, d, composition, high_to_low)


def test_4_8() -> None:
    # 8 polynomials (columns), each of length 2⁵, so over 5 variables.
    v = 4
    d = 8
    multidegree = tuple(1 if x % 5 == 0 else 0 for x in range(d))
    composition = Polynomial(Elem128b, d, {multidegree: Elem128b.one()})  # product monomial on d variables.
    for high_to_low in [True, False]:
        run_test(v, d, composition, high_to_low)


@pytest.mark.slow
def test_5_7() -> None:
    # 5 polynomials, each of length 2⁷, so over 7 variables.
    v = 7
    d = 5
    composition = Polynomial(Elem128b, d, {tuple([1] * d): Elem128b.one()})  # product monomial on d variables.
    for high_to_low in [True, False]:
        run_test(v, d, composition, high_to_low)


@pytest.mark.slow
def test_3_5_other_composition() -> None:
    # 3 polynomials, each of length 2⁵, so over 5 variables.
    v = 5
    d = 3
    composition = Polynomial(
        Elem128b,
        d,
        {
            (0, 0, 0): Elem128b(int("0xaeaeaeaeaeaeaeaebdbdbdbdbdbdbdbd", 16)),
            (1, 2, 3): Elem128b.one(),
        },
    )  # 3-variate polynomial: X₀ ⋅ X₁² ⋅ X₂³ + 0xaeaeaeaeaeaeaeaebdbdbdbdbdbdbdbd
    for high_to_low in [True, False]:
        run_test(v, d, composition, high_to_low)


@pytest.mark.slow
def test_3_5_crazy() -> None:
    v = 5
    d = 3
    composition = Polynomial(
        Elem128b,
        d,
        {
            (0, 0, 0): Elem128b(int("0xaeaeaeaeaeaeaeaebdbdbdbdbdbdbdbd", 16)),
            (1, 2, 4): Elem128b(int("0xaaee", 16)),
            (2, 3, 10): Elem128b.one(),
        },
    )
    for high_to_low in [True, False]:
        run_test(v, d, composition, high_to_low)
