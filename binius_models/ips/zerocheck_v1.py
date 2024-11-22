import math

from binius_models.ips.utils import (
    Elem128b,
    Polynomial128,
    evaluate_univariate,
    inverse_matrix,
    linearly_interpolate,
    mul_matrix_vec,
    multilinear_query,
    vandermonde,
)


class Zerocheck:
    def __init__(
        self,
        multilinears: list[list[Elem128b]],
        composition: Polynomial128,
        zerocheck_challenges: list[Elem128b],
        switchover: int,
    ) -> None:
        # polynomials: a list of _multilinear_ polynomials, of equal sizes.
        # we're going to run sumcheck, where g is the product of the given multilinear polynomials.
        # each is given as a list of coefficients in the Lagrange basis, where these coefficients are 𝔽₂-elements
        self.multilinears = multilinears
        self.composition = composition
        length = len(multilinears[0])
        self.v = int(math.log2(length))
        assert length == 1 << self.v  # length is a power of 2
        assert all(len(multilinear) == length for multilinear in multilinears[1:])  # multilinears are of equal lengths
        assert composition.variables == len(multilinears)  # number of multilinears is same as num vars of composition
        self.round = 0
        self.switchover = switchover
        self.tensor = [Elem128b.one()] + [Elem128b.zero()] * ((1 << self.switchover) - 1)

        # Fields specific to zerocheck prover
        self.zerocheck_challenges = zerocheck_challenges
        assert len(zerocheck_challenges) == self.v - 1
        self.eq_ind = multilinear_query(self.zerocheck_challenges)
        self.interpolation_matrix = inverse_matrix(vandermonde(self.composition.degree + 1))

    def compute_round_polynomial(self) -> list[Elem128b]:
        # computes the first half of the round: everything up until the round polynomial is sent to the verifier.
        # returns the list of evaluations of {g_{self.round}(k)) at the points [1, ..., d]; length is d
        assert self.round in range(self.v)

        # Vector of evals.
        # Omit evaluation at 0 in all rounds.
        # In round 0 no need to evaluate at 1.
        evaluations = [Elem128b.zero()] * (self.composition.degree if self.round > 0 else self.composition.degree - 1)
        round_v = self.v - self.round

        # Update `eq_ind` for the current round, folding it in half.
        if self.round > 0:
            current_evals = self.eq_ind[:]
            self.eq_ind = [(current_evals[i * 2] + current_evals[i * 2 + 1]) for i in range(len(current_evals) // 2)]

        # `folds` represents the evaluations of the partially evaluated multilinear polynomials over the `round_v`
        # dimensional boolean hypercube. The partial evaluation amounts to specializing the `self.round` least
        # significant variables to the received verifier round challenges. In the initial round and after switchover,
        # we have these evaluations stored. Otherwise, they must be computed via a tensor MAC.

        # NB: Populating `folds` is always necessary when the composition degree is at least two. We will need to
        # evaluate each round polynomial at `X = 2`. This requires us to linearly extrapolate for each multilinear `f`
        # and for each `(round_v-1)` dimensional hypercube point `x`. Specifically we will extrapolate `f(r, 0, x)` and
        # `f(r, 1, x)` to `f(r, 2, x)` where `r` is the vector of verifier round challenges received thus far.
        folds = [[Elem128b.zero() for _ in range(1 << round_v)] for _ in range(self.composition.variables)]

        for i in range(1 << round_v - 1):
            for j in range(self.composition.variables):
                for k in range(2):  # first, let's populate the first two rows...
                    idx = i << 1 | k
                    if self.round > self.switchover or self.round == 0:  # just copy `multilinears`; do nothing to it.
                        folds[j][idx] = self.multilinears[j][idx]  # no-op, do just copy.
                    else:
                        folds[j][idx] = Elem128b.zero()
                        for h in range(1 << self.round):
                            folds[j][idx] += self.tensor[h] * self.multilinears[j][idx << self.round | h]

        for i in range(1 << round_v - 1):
            arguments = [[folds[j][i << 1 | k] for j in range(self.composition.variables)] for k in range(2)]
            eq_ind_factor = self.eq_ind[i]

            if self.round > 0:
                evaluations[0] += self.composition.evaluate(arguments[1]) * eq_ind_factor

            for k in range(2, self.composition.degree + 1):
                argument = [  # note: each Elem128b(k) below will live in a small field! implement it as such.
                    linearly_interpolate((arguments[0][j], arguments[1][j]), Elem128b(k))
                    for j in range(self.composition.variables)
                ]
                composite_value = self.composition.evaluate(argument)
                value = composite_value * eq_ind_factor
                if self.round == 0:
                    evaluations[k - 2] += value
                else:
                    evaluations[k - 1] += value

        if self.round == self.switchover:
            self.multilinears = folds
        return evaluations

    def receive_challenge(self, r: Elem128b) -> None:
        round_v = self.v - self.round
        if self.round < self.switchover:  # expand tensor
            for h in range(1 << self.round):
                idx0 = h
                idx1 = idx0 | 1 << self.round
                self.tensor[idx1] = self.tensor[idx0] * r
                self.tensor[idx0] -= self.tensor[idx1]
        else:  # classical folding
            for j in range(self.composition.variables):
                self.multilinears[j] = [
                    linearly_interpolate((self.multilinears[j][i << 1], self.multilinears[j][i << 1 | 1]), r)
                    for i in range(1 << round_v - 1)
                ]
        self.round += 1

    # Corresponds to `EvaluationDomain::interpolate()`.
    def interpolate(self, evaluations: list[Elem128b]) -> list[Elem128b]:
        # interpreting "evaluations" as the evals of an unknown polynomial of degree ≤ d at the points (0, ..., d),
        # returns the evaluation of said polynomial at `point`. uses barycentric extrapolation.
        # variant w/o division: ref: https://gitlab.com/ulvetanna/frobenius/-/blob/main/src/polynomial/univariate.rs
        assert len(evaluations) == self.composition.degree + 1
        return mul_matrix_vec(self.interpolation_matrix, evaluations)

    # Checks that every row evaluates to zero.
    def check_validity(self) -> bool:
        valid = True
        for h in range(1 << self.v):
            if Elem128b.zero() != self.composition.evaluate([multilinear[h] for multilinear in self.multilinears]):
                valid = False
        return valid

    # Corresponds to the first part of `reduce_intermediate_round_claim_helper()`.
    def restore_constant_term(self, coeffs: list[Elem128b], current_round_sum: Elem128b):
        if self.round == 0:
            assert current_round_sum == Elem128b.zero()
            expected_linear_term = Elem128b.zero() - sum(coeffs[1:], Elem128b.zero())
            assert coeffs[0] == expected_linear_term
            return [Elem128b.zero()] + coeffs[:]
        else:
            constant_term = current_round_sum - self.zerocheck_challenges[self.round - 1] * sum(coeffs, Elem128b.zero())
            return [constant_term] + coeffs[:]

    def round_evals_to_coeffs(self, evaluations: list[Elem128b], current_round_sum: Elem128b) -> list[Elem128b]:
        if self.round == 0:
            # Corresponds to `ZerocheckFirstRoundEvaluator::round_evals_to_coeffs()`.
            evaluations.insert(0, Elem128b.zero())
            evaluations.insert(0, Elem128b.zero())
        else:
            # Corresponds to `ZerocheckLaterRoundEvaluator::round_evals_to_coeffs()`.
            alpha_i = self.zerocheck_challenges[self.round - 1]
            v = (current_round_sum - evaluations[0] * alpha_i) / (Elem128b.one() - alpha_i)
            evaluations.insert(0, v)
        coeffs = self.interpolate(evaluations)[1:]
        return coeffs

    def verify(self, rounds: list[list[Elem128b]], eval_point: list[Elem128b]) -> Elem128b:
        current_round_sum = Elem128b.zero()
        for r in range(len(rounds)):
            self.round = r
            coeffs = rounds[r]
            challenge = eval_point[r]
            if r == 0:
                assert current_round_sum == Elem128b.zero()
            coeffs = self.restore_constant_term(coeffs, current_round_sum)
            current_round_sum = evaluate_univariate(coeffs, challenge)
        return current_round_sum
