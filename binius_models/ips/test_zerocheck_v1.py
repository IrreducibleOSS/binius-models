from .utils import (
    Elem8b,
    Elem128b,
    Polynomial128,
    compute_switchover,
    evaluate_univariate,
)
from .zerocheck_v1 import Zerocheck


def run_test(v, d, composition):
    multilinears = [
        [Elem128b(Elem8b.random().value) if i % d != j else Elem128b.zero() for i in range(1 << v)] for j in range(d)
    ]
    zerocheck_challenges = [Elem128b.random() for _ in range(v - 1)]
    zerocheck = Zerocheck(multilinears, composition, zerocheck_challenges, compute_switchover(v, d, 100))

    rounds = []
    assert zerocheck.check_validity()
    current_round_sum = Elem128b.zero()
    eval_point = []
    for r in range(zerocheck.v):
        evaluations = zerocheck.compute_round_polynomial()
        coeffs = zerocheck.round_evals_to_coeffs(evaluations, current_round_sum)
        rounds.append(coeffs[:])
        coeffs = zerocheck.restore_constant_term(coeffs, current_round_sum)
        challenge = Elem128b.random()
        eval_point.append(challenge)
        zerocheck.receive_challenge(challenge)
        current_round_sum = evaluate_univariate(coeffs, challenge)

    # Use the randomness from `eval_point` to make the randomness repeatable for the verification.
    final_sum = zerocheck.verify(rounds, eval_point)
    assert current_round_sum == final_sum

    print("eval_point: ", eval_point)
    print("eval: ", final_sum)


def test_1_5() -> None:
    # 1 polynomial! of length 2⁵, so over 5 variables.
    v = 5
    d = 1
    composition = Polynomial128(d, {tuple([1] * d): Elem128b.one()})  # identity polynomial on 1 variable
    run_test(v, d, composition)


def test_3_5() -> None:
    # 3 polynomials, each of length 2⁵, so over 5 variables.
    v = 5
    d = 3
    composition = Polynomial128(d, {tuple([1] * d): Elem128b.one()})  # product monomial on d variables.
    run_test(v, d, composition)
