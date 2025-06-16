import pytest

from .ring_switching import (
    LargeFieldElem,
    RingSwitching,
    SmallFieldElem,
    evaluate_multilinear,
    large_large_dot_product,
    tensor_expansion,
)
from .tensor_alg import TensorAlgElem


# test that the ring-switched FRI-Binius works correctly.
# in particular, we instantiate a FRI-Binius instance from the RingSwitching protocol
# and then run the FRI-Binius protocol itself.
@pytest.mark.parametrize("var, log_inv_rate", [(5, 1), (5, 2), (8, 1)])
def test_ring_switched_pcs(var: int, log_inv_rate: int) -> None:
    multilinear = [SmallFieldElem.random() for _ in range(1 << var)]

    ring_switch = RingSwitching(var, log_inv_rate)
    ring_switch.commit(multilinear)

    # a random element of L^{var}, at which we want to evaluate the polynomial.
    evaluation_point = [LargeFieldElem.random() for _ in range(var)]
    r_double_prime = [LargeFieldElem.random() for _ in range(ring_switch.kappa)]
    s_hat = ring_switch.initialize_proof(evaluation_point, r_double_prime)

    # begin verifier's first check: the ŝ vs. s column combination check
    # in order to actually test this, we need to independently compute what s should be. do that now.
    s = evaluate_multilinear(multilinear, evaluation_point)
    tensor_expanded_first_chunk_input = tensor_expansion(evaluation_point[: ring_switch.kappa])
    assert s == large_large_dot_product(tensor_expanded_first_chunk_input, s_hat.column_representation())

    # prepare beginning sumcheck. first compute s₀ := row-batch of ŝ.
    tensor_expanded_r_double_prime = tensor_expansion(r_double_prime)
    s = large_large_dot_product(s_hat.row_representation(), tensor_expanded_r_double_prime)  # === s₀ in the paper.

    e = TensorAlgElem.one()
    for i in range(ring_switch.fri_binius.var):  # proceed with actual sumcheck
        evaluations = ring_switch.advance_state()
        evaluations.insert(0, evaluations[0] + s)
        challenge = LargeFieldElem.random()
        ring_switch.receive_challenge(challenge)
        s = ring_switch.sumcheck.interpolate(evaluations, challenge)
        e += e.mul_by_phi_0(evaluation_point[ring_switch.kappa + i]) + e.mul_by_phi_1(challenge)
    s_prime = ring_switch.finalize()

    assert s == s_prime * large_large_dot_product(e.row_representation(), tensor_expanded_r_double_prime)

    # begin final stage; i.e., actually run the sub-FRI-Binius. this is copied from test_fri_binius
    s = s_prime
    one = LargeFieldElem.one()
    eq_r_rprime = one
    for i in range(ring_switch.fri_binius.var):
        evaluations = ring_switch.fri_binius.advance_state()
        evaluations.insert(0, evaluations[0] + s)
        challenge = LargeFieldElem.random()
        ring_switch.fri_binius.receive_challenge(challenge)
        s = ring_switch.fri_binius.sumcheck.interpolate(evaluations, challenge)
        eq_r_rprime *= (ring_switch.challenges[i] + one) * (challenge + one) + ring_switch.challenges[i] * challenge
    c = ring_switch.fri_binius.finalize()
    assert s == c * eq_r_rprime

    num_queries = 100  # say
    for _ in range(num_queries):
        ring_switch.fri_binius.verifier_query(c)
