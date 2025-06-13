import math

from binius_models.ips.utils import (
    Elem128b,
    Polynomial128,
    compute_switchover,
    linearly_interpolate,
)

from ..utils.utils import is_power_of_two


class Sumcheck:
    """
    multilinears: a list of _multilinear_ polynomials, of equal sizes.

    high_to_low: Is the data represented in first-variable major (experimental) order?.
    In high-to-low order, positions being folded together are half the multilinear apart instead of adjacent.

    we're going to run sumcheck, where g is the product of the given multilinear polynomials.
    each is given as a list of coefficients in the Lagrange basis, where these coefficients are 𝔽₂-elements
    """

    def __init__(
        self, multilinears: list[list[Elem128b]], composition: Polynomial128, high_to_low: bool = False
    ) -> None:
        length = len(multilinears[0])
        assert is_power_of_two(length)
        assert all(len(multilinear) == length for multilinear in multilinears)  # multilinears are of equal lengths
        assert composition.variables == len(multilinears)  # number of multilinears is same as num vars of composition
        self.v = int(math.log2(length))
        self.multilinears = multilinears
        self.composition = composition
        self.round = 0
        self.switchover = compute_switchover(self.v, composition.degree, 100)
        self.tensor = [Elem128b.one()] + [Elem128b.zero()] * ((1 << self.switchover) - 1)
        self._precompute_barycentric_weights()
        self.high_to_low = high_to_low

    def _precompute_barycentric_weights(self) -> None:
        # begin precomputation of barycentric constants; see https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
        # will only be used by the verifier, to extrapolate
        self.w = []  # the degree of the polynomial we will need to extrapolate will be composition.degree.
        for i in range(self.composition.degree + 1):
            product = Elem128b.one()
            for j in range(self.composition.degree + 1):
                if j == i:
                    continue
                product *= Elem128b(i ^ j)
            self.w.append(product.inverse())

    def compute_round_polynomial(self) -> list[Elem128b]:
        # computes the first half of the round: everything up until the round polynomial is sent to the verifier.
        # returns the list of evaluations of {g_{self.round}(k)) at the points [1, ..., d]; length is d
        assert self.round in range(self.v)

        evaluations = [Elem128b.zero()] * self.composition.degree  # vector of evals; omit evaluation at 0.
        round_v = self.v - self.round
        folds = [[Elem128b.zero() for _ in range(1 << round_v)] for _ in range(self.composition.variables)]

        for i in range(1 << round_v - 1):
            for j in range(self.composition.variables):
                for k in range(2):  # first, let's populate the first two rows...
                    idx = i << 1 | k
                    if self.round > self.switchover or self.round == 0:  # just copy `multilinears`; do nothing to it.
                        folds[j][idx] = self.multilinears[j][idx]  # no-op, do just copy.
                    else:
                        if self.high_to_low:
                            folds[j][idx] = sum(
                                (
                                    self.tensor[h << self.switchover - self.round]
                                    * self.multilinears[j][h << round_v | idx]
                                    for h in range(1 << self.round)
                                ),
                                Elem128b.zero(),  # NOTE: this whole thing above is a base * extension dot product.
                            )
                        else:
                            folds[j][idx] = sum(
                                (
                                    self.tensor[h] * self.multilinears[j][idx << self.round | h]
                                    for h in range(1 << self.round)
                                ),
                                Elem128b.zero(),  # NOTE: this whole thing above is a base * extension dot product.
                            )
                    # which indices into self.multilinears[j] do we care about? the index we want is as follows:
                    # [ _____i (v - 1 - self.round bits)_____  ||  _k (1 bit)_  ||  _____h (self.round bits)_____ ]
                    # you can see that the total bit-width of our index is v bits, which is what we want.

        for i in range(1 << round_v - 1):
            # some shorthand; there is no new computation here...
            if self.high_to_low:
                arguments = [
                    [folds[j][k << round_v - 1 | i] for j in range(self.composition.variables)] for k in range(2)
                ]
            else:
                arguments = [[folds[j][i << 1 | k] for j in range(self.composition.variables)] for k in range(2)]
            evaluations[0] += self.composition.evaluate(arguments[1])
            for k in range(2, self.composition.degree + 1):
                argument = [  # note: each Elem128b(k) below will live in a small field! implement it as such.
                    linearly_interpolate((arguments[0][j], arguments[1][j]), Elem128b(k))
                    for j in range(self.composition.variables)
                ]
                evaluations[k - 1] += self.composition.evaluate(argument)

        if self.round == self.switchover:
            self.multilinears = folds
        return evaluations

    def receive_challenge(self, r: Elem128b) -> None:
        round_v = self.v - self.round
        if self.round < self.switchover:  # expand tensor
            if self.high_to_low:
                for h in range(1 << self.round):
                    idx0 = h << self.switchover - self.round
                    idx1 = idx0 | 1 << self.switchover - self.round - 1
                    self.tensor[idx1] = self.tensor[idx0] * r
                    self.tensor[idx0] -= self.tensor[idx1]
            else:
                for h in range(1 << self.round):
                    idx0 = h
                    idx1 = idx0 | 1 << self.round
                    self.tensor[idx1] = self.tensor[idx0] * r
                    self.tensor[idx0] -= self.tensor[idx1]
        else:  # classical folding
            for j in range(self.composition.variables):
                if self.high_to_low:
                    self.multilinears[j] = [
                        linearly_interpolate((self.multilinears[j][i], self.multilinears[j][1 << round_v - 1 | i]), r)
                        for i in range(1 << round_v - 1)
                    ]
                else:
                    self.multilinears[j] = [
                        linearly_interpolate((self.multilinears[j][i << 1], self.multilinears[j][i << 1 | 1]), r)
                        for i in range(1 << round_v - 1)
                    ]
        self.round += 1

    def interpolate(self, evaluations: list[Elem128b], point: Elem128b) -> Elem128b:
        # interpreting "evaluations" as the evals of an unknown polynomial of degree ≤ d at the points (0, ..., d),
        # returns the evaluation of said polynomial at `point`. uses barycentric extrapolation.
        # variant w/o division: ref: https://gitlab.com/ulvetanna/frobenius/-/blob/main/src/polynomial/univariate.rs
        assert len(evaluations) == self.composition.degree + 1

        terms_partial_prod = Elem128b.one()
        result = Elem128b.zero()
        for i in range(self.composition.degree + 1):
            term = point + Elem128b(i)
            result *= term
            result += evaluations[i] * self.w[i] * terms_partial_prod
            terms_partial_prod *= term
        return result

    def query(self) -> Elem128b:
        assert self.round == self.v
        return self.composition.evaluate([multilinear[0] for multilinear in self.multilinears])

    def sum(self) -> Elem128b:
        # helper method: returns the actual statement that we're proving;
        # namely, it's the actual sum of g over the entire cube.
        return sum(
            (
                self.composition.evaluate([multilinear[h] for multilinear in self.multilinears])
                for h in range(1 << self.v)
            ),
            Elem128b.zero(),
        )
