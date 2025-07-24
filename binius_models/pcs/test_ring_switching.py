import pytest

from ..finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField
from ..finite_fields.utils import tensor_expand
from .ring_switching import RingSwitching
from .tensor_alg import TensorAlgElem


class Elem8bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(3)


class Elem128bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


# this tests the "separated" variant of ring-switching, which is oblivious to which large-field PCS it uses.
@pytest.mark.parametrize("var, log_inv_rate", [(5, 1), (5, 2), (8, 1)])
def test_ring_switching(var: int, log_inv_rate: int) -> None:
    multilinear = [Elem8bFP.random() for _ in range(1 << var)]
    ring_switch = RingSwitching(Elem8bFP, Elem128bFP, var, log_inv_rate)
    ring_switch.commit(multilinear)
    r = [Elem128bFP.random() for _ in range(var)]

    # in order to actually test this, we need to independently precompute what s should be. do that now.
    r_tensor = tensor_expand(Elem128bFP, r, var)
    s = sum((r_tensor[i] * multilinear[i].upcast(Elem128bFP) for i in range(1 << var)), Elem128bFP.zero())

    # begin actual test in earnest
    s_hat = ring_switch.s_hat(r)
    r_low_tensor = tensor_expand(Elem128bFP, r[: ring_switch.kappa], ring_switch.kappa)
    assert s == sum((s_hat.columns[i] * r_low_tensor[i] for i in range(1 << ring_switch.kappa)), Elem128bFP.zero())

    r_double_prime = [Elem128bFP.random() for _ in range(ring_switch.kappa)]
    ring_switch.receive_r_double_prime(r_double_prime)
    r_double_prime_tensor = tensor_expand(Elem128bFP, r_double_prime, ring_switch.kappa)

    temp = s_hat.transpose().columns
    s = sum((temp[i] * r_double_prime_tensor[i] for i in range(1 << ring_switch.kappa)), Elem128bFP.zero())
    e = TensorAlgElem(Elem8bFP, Elem128bFP, [Elem128bFP.one()] + [Elem128bFP.zero()] * ((1 << ring_switch.kappa) - 1))
    for i in range(ring_switch.l_prime):  # proceed with actual sumcheck
        evaluations = ring_switch.sumcheck.compute_round_polynomial()
        evaluations.insert(0, evaluations[0] + s)  # s₀ = s
        challenge = Elem128bFP.random()
        ring_switch.sumcheck.receive_challenge(challenge)
        s = ring_switch.sumcheck.interpolate(evaluations, challenge)
        e += e.scale_vertical(r[ring_switch.kappa + i]) + e.scale_horizontal(challenge)
    s_prime = ring_switch.finalize()
    temp = e.transpose().columns
    assert (
        s
        == sum((r_double_prime_tensor[i] * temp[i] for i in range(1 << ring_switch.kappa)), Elem128bFP.zero()) * s_prime
    )

    # begin final stage; i.e., actually run the sub-FRI-Binius. this is copied from test_fri_binius
    s = s_prime
    one = Elem128bFP.one()
    eq_r_rprime = one
    for i in range(ring_switch.fri_binius.var):
        evaluations = ring_switch.fri_binius.advance_state()
        evaluations.insert(0, evaluations[0] + s)
        challenge = Elem128bFP.random()
        ring_switch.fri_binius.receive_challenge(challenge)
        s = ring_switch.fri_binius.sumcheck.interpolate(evaluations, challenge)
        eq_r_rprime *= one + ring_switch.sumcheck.challenges[i] + challenge
    c = ring_switch.fri_binius.finalize()
    assert s == c * eq_r_rprime

    num_queries = 100  # say
    for _ in range(num_queries):
        ring_switch.fri_binius.verifier_query(c)
