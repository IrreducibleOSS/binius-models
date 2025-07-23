import pytest

from binius_models.finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField

from .batched_fri_binius import BatchedFRIBinius, SumcheckClaim


class Elem128b(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


@pytest.mark.parametrize("sizes, log_inv_rate", [((4,), 1), ((3, 2, 2, 1), 1), ((3, 3, 1), 1), ((6, 4, 4, 2, 1), 2)])
def test_batched_fri_binius(sizes, log_inv_rate):
    claims = []
    for i, size in enumerate(sizes):
        assert size > 0  # constant (0-variate) multilinears are not allowed.
        multilinear = [Elem128b.random() for _ in range(1 << size)]
        eq_mock = [Elem128b.random() for _ in range(1 << size)]  # let them both be random for now
        claim = SumcheckClaim(Elem128b, multilinear, eq_mock, i)
        claims.append(claim)

    batched_fri_binius = BatchedFRIBinius(Elem128b, log_inv_rate, claims)
    positions = batched_fri_binius.positions
    batched_fri_binius.commit()

    batching_randomness = Elem128b.random()

    s = Elem128b.zero()  # this is the value of the master batched sumcheck claim.
    randomness = Elem128b.one()
    for claim in sorted(claims, key=lambda claim: claim.sumcheck.v):
        s += randomness * claim.value  # right now, value is being computed; really, it will "come to us" as a claim.
        randomness *= batching_randomness

    batched_fri_binius.initialize(batching_randomness)

    randomness = Elem128b.one()  # start over
    for i in range(batched_fri_binius.var):
        round_polynomial = batched_fri_binius.advance_state()
        round_polynomial.insert(0, round_polynomial[0] + s)
        challenge = Elem128b.random()
        s = batched_fri_binius.manager.claims[0].sumcheck.interpolate(round_polynomial, challenge)

        evaluations = batched_fri_binius.receive_challenge(challenge)
        assert len(evaluations) == positions[i] - positions[i + 1]
        for j in range(positions[i + 1], positions[i]):
            claim = claims[j]
            evaluation = evaluations[j - positions[i + 1]]  # this + the above is really just a zip in disguise
            s -= randomness * evaluation * claim.sumcheck.multilinears[1][0]  # prune off sumcheck component
            # WHAT'S GOING ON HERE? claim.multilinears[1] is a dummy mock for the eq_indicator.
            # in practice we will compute its value at (r₀, … , rᵢ) ourselves, using a succinct local computation.
            # instead, we lift it out of claim, using some work the prover has just completed.
            # evaluation of course is their nondeterministically claimed value for the actual value of the multilinear.
            randomness *= batching_randomness
            claim.evaluation = evaluation  # some datastructure business, for the verifier to store this locally.
        j = positions[i]  # exactly the "end" of the above loop
        while j < len(claims):
            claim = claims[j]
            zeroth = claim.evaluation
            first = Elem128b.zero()
            if claim.next < len(claims):
                first = claims[claim.next].evaluation
                claim.next = claims[claim.next].next  # skip the next guy!!!
            claim.evaluation = zeroth + challenge * (first - zeroth)
            j = claim.next
        assert j == len(claims)  # debug assert; only for sanity; verifier doesn't actually need to check

    assert claims[0].next == len(claims)  # this is a "debug assert"; should always hold (even for malicious prover)
    c = batched_fri_binius.finalize()  # grab final fri value.
    assert c == claims[0].evaluation  # this assertion is actually part of what the verifier MUST check
    assert s == Elem128b.zero()  # same for this one

    num_queries = 100  # say
    for _ in range(num_queries):
        batched_fri_binius.verifier_query(c)
