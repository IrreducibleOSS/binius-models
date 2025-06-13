import pytest

from .fri_binius import Elem128b, FRIBinius


@pytest.mark.parametrize("var, log_inv_rate", [(4, 1), (5, 2)])
def test_fri_binius(var, log_inv_rate):
    zero = Elem128b.zero()
    one = Elem128b.one()

    multilinear = [Elem128b.random() for _ in range(1 << var)]
    fri_binius = FRIBinius(Elem128b, var, log_inv_rate)
    fri_binius.commit(multilinear)

    evaluation_point = [Elem128b.random() for _ in range(var)]
    eq_r = [one] + [zero] * (len(multilinear) - 1)  # actually evaluate it ourselves, to test
    for i in range(fri_binius.var):  # expand tensor of eval point, just to get claim for ourselves, to check.
        for h in range(1 << i):  # in practice this would be sent to us by the prover, as the statement.
            eq_r[1 << i | h] = eq_r[h] * evaluation_point[i]
            eq_r[h] -= eq_r[1 << i | h]
    s = sum((multilinear[i] * eq_r[i] for i in range(len(multilinear))), zero)

    fri_binius.initialize_proof(evaluation_point)

    eq_r_rprime = one
    for i in range(fri_binius.var):
        evaluations = fri_binius.advance_state()
        evaluations.insert(0, evaluations[0] + s)
        challenge = Elem128b.random()
        fri_binius.receive_challenge(challenge)
        s = fri_binius.sumcheck.interpolate(evaluations, challenge)
        eq_r_rprime *= (evaluation_point[i] + one) * (challenge + one) + evaluation_point[i] * challenge
    c = fri_binius.finalize()
    assert s == c * eq_r_rprime

    num_queries = 100  # say
    for _ in range(num_queries):
        fri_binius.verifier_query(c)
