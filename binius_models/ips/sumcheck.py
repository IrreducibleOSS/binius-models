import math
from typing import Generic, TypeVar

from binius_models.finite_fields.finite_field import FiniteFieldElem
from binius_models.ips.polynomial import Polynomial

from ..utils.utils import is_power_of_two

F = TypeVar("F", bound=FiniteFieldElem)


def compute_switchover(v: int, d: int, ratio: int) -> int:
    # returns: at which point should we switch from b × e muls, plus tensor expansion, to classical-style folding?
    # parameter `ratio`: what is the cost of an e × e mult, relative to that of a b × e?

    # in the simplest case len(multilinears) == 1 and subfield size 1 bit (and ignoring XORs), the best is at v/2.
    # more generally, allowing len(multilinears) == d you can calculate that the breakeven point happens at exactly
    # (log(d) + v) / 2. why? for each round, the cost of self.receive_challenge is:
    # - if round < switchover: exactly 2ʳᵒᵘⁿᵈ e × e mults.
    # - if round ≥ switchover, exactly d ⋅ 2ᵛ⁻ʳᵒᵘⁿᵈ e × e mults.
    # thus we're interested in the smallest r for which d ⋅ 2ᵛ⁻ʳ ≤ 2ʳ first becomes true.
    # taking logs of both sides, we see the inequality log(d) + v ≤ 2 ⋅ r, and we're done.

    # it becomes more complicated if we want to count the cost of b × e mults---
    # as we might, say, as soon as the field size of the polys is > 1 bit, or if we care about the cost of XORs.
    # of course the higher the cost of b × es, the earlier we will want the switchover to actually happen.
    # indeed, now counting b × es, the total cost of each round is 2ᵛ⁻ʳᵒᵘⁿᵈ⁻¹ ⋅ deg polynomial evaluations,
    # where write `deg` for the total degree of the composition polynomial (it equals d when it's just a product),
    # (this cost is there regardless of pre-switchover or not), PLUS, in addition:
    # - if round < switchover: d ⋅ 2ᵛ b × e (compute folds on the fly) + 2ʳᵒᵘⁿᵈ e × e (tensor expand)
    # - if round > switchover: d ⋅ 2ᵛ⁻ʳᵒᵘⁿᵈ e × e (classical folding)
    # - if round == switchover: d ⋅ 2ᵛ b × e + d ⋅ 2ᵛ⁻ʳᵒᵘⁿᵈ e × e (on-the-fly and fold).
    # when you work it out, for arbitrary switchover r ∈ {0, ..., v - 1}, the total cost across all rounds becomes:
    # f(r) :=
    # (2ʳ − 1) e × e (total work of tensor expansion, up to but excluding the switchover round)
    # + d ⋅ r ⋅ 2ᵛ b × e (total work of only-the-fly folds, including switchover round, but excluding 0th round)
    # + d ⋅ (2ᵛ⁻ʳ⁺¹ - 2) e × e (total work of classical folds, including the switchover round and following rounds)
    # so our goal is to choose r ∈ {0, ..., v - 1} so as to minimze f(r), which is a straightforward calculation.
    # though the minimal value in practice will depend on the relative cost of b × es versus e × es, as well as d.
    # you can see that we have an amount of b × es which increases linearly in r (penalty for delaying the switch),
    # as well as the familiar exponentially increasing and decreasing costs in e × es of tensor and folding, resp.
    return min(range(v), key=lambda r: ((1 << r) - 1) * ratio + d * r * (1 << v) + d * ((1 << v - r + 1) - 2) * ratio)


# Corresponds to `binius_core::polynomial::extrapolate_line()`.
def linearly_interpolate(points: tuple[F, F], r: F) -> F:
    return points[0] + (points[1] - points[0]) * r


class Sumcheck(Generic[F]):
    """
    multilinears: a list of _multilinear_ polynomials, of equal sizes.

    high_to_low: Is the data represented in first-variable major (experimental) order?.
    In high-to-low order, positions being folded together are half the multilinear apart instead of adjacent.

    we're going to run sumcheck, where g is the product of the given multilinear polynomials.
    each is given as a list of coefficients in the Lagrange basis, where these coefficients are 𝔽₂-elements
    """

    def __init__(
        self, field: type[F], multilinears: list[list[F]], composition: Polynomial[F], high_to_low: bool = False
    ) -> None:
        length = len(multilinears[0])
        assert is_power_of_two(length)
        assert all(len(multilinear) == length for multilinear in multilinears)  # multilinears are of equal lengths
        assert composition.variables == len(multilinears)  # number of multilinears is same as num vars of composition
        self.field = field
        self.v = int(math.log2(length))
        self.multilinears = multilinears
        self.composition = composition
        self.challenges: list[F] = []
        # you can get surprisingly far without memoizing these. it turns out they wind up being useful in some cases.
        # e.g., in fribinius, etc., we need them. go ahead and memoize them here, for use by consuming applications.
        self.round = 0
        self.switchover = compute_switchover(self.v, composition.degree, 100)
        self.tensor = [self.field.one()] + [self.field.zero()] * ((1 << self.switchover) - 1)
        self._precompute_barycentric_weights()
        self.high_to_low = high_to_low

    def _precompute_barycentric_weights(self) -> None:
        # begin precomputation of barycentric constants; see https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
        # will only be used by the verifier, to extrapolate
        self.w = []  # the degree of the polynomial we will need to extrapolate will be composition.degree.
        for i in range(self.composition.degree + 1):
            product = self.field.one()
            for j in range(self.composition.degree + 1):
                if j == i:
                    continue
                product *= self.field(i ^ j)
            self.w.append(product.inverse())

    def compute_round_polynomial(self) -> list[F]:
        # computes the first half of the round: everything up until the round polynomial is sent to the verifier.
        # returns the list of evaluations of {g_{self.round}(k)) at the points [1, ..., d]; length is d
        assert self.round in range(self.v)

        evaluations = [self.field.zero()] * self.composition.degree  # vector of evals; omit evaluation at 0.
        round_v = self.v - self.round
        folds = [[self.field.zero() for _ in range(1 << round_v)] for _ in range(self.composition.variables)]

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
                                self.field.zero(),  # NOTE: this whole thing above is a base * extension dot product.
                            )
                        else:
                            folds[j][idx] = sum(
                                (
                                    self.tensor[h] * self.multilinears[j][idx << self.round | h]
                                    for h in range(1 << self.round)
                                ),
                                self.field.zero(),  # NOTE: this whole thing above is a base * extension dot product.
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
                argument = [  # note: each self.field(k) below will live in a small field! implement it as such.
                    linearly_interpolate((arguments[0][j], arguments[1][j]), self.field(k))
                    for j in range(self.composition.variables)
                ]
                evaluations[k - 1] += self.composition.evaluate(argument)

        if self.round == self.switchover:
            self.multilinears = folds
        return evaluations

    def receive_challenge(self, r: F) -> None:
        self.challenges.append(r)
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

    def interpolate(self, evaluations: list[F], point: F) -> F:
        # interpreting "evaluations" as the evals of an unknown polynomial of degree ≤ d at the points (0, ..., d),
        # returns the evaluation of said polynomial at `point`. uses barycentric extrapolation.
        # variant w/o division: ref: https://gitlab.com/ulvetanna/frobenius/-/blob/main/src/polynomial/univariate.rs
        assert len(evaluations) == self.composition.degree + 1

        terms_partial_prod = self.field.one()
        result = self.field.zero()
        for i in range(self.composition.degree + 1):
            term = point + self.field(i)
            result *= term
            result += evaluations[i] * self.w[i] * terms_partial_prod
            terms_partial_prod *= term
        return result

    def query(self) -> F:
        assert self.round == self.v
        return self.composition.evaluate([multilinear[0] for multilinear in self.multilinears])

    def sum(self) -> F:
        # helper method: returns the actual statement that we're proving;
        # namely, it's the actual sum of g over the entire cube.
        return sum(
            (
                self.composition.evaluate([multilinear[h] for multilinear in self.multilinears])
                for h in range(1 << self.v)
            ),
            self.field.zero(),
        )
