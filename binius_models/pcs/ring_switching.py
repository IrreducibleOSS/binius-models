# (C) 2024 Irreducible Inc.

# An Implementation of Ring-Switching protocol.
# On a high level, this allows us to instantiate a "small field" PCS given a large-field PCS with no extra overhead.

# After one shot, one is reduced to a multilinear evaluation claim over
# A := L ⊗_K L. We then run a sort of induced "row-batched" sumcheck.
# The source for this material is Sections 4 and 5 of [DP24], which has as antecedent [D24].

# A word on notation. In [DP24], ŝ is the A-valued evaluation
# of the multilinear and s₀ is the row-batched version of this claim. We follow this notation.

from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem
from ..finite_fields.utils import tensor_expand
from ..ips.polynomial import Polynomial
from ..ips.sumcheck import Sumcheck
from .fri_binius import FRIBinius
from .tensor_alg import TensorAlgElem

K = TypeVar("K", bound=FiniteFieldElem)
L = TypeVar("L", bound=FiniteFieldElem)


class RingSwitching(Generic[K, L]):
    def __init__(self, small: type[K], large: type[L], var: int, log_inv_rate: int) -> None:
        self.small = small
        self.large = large
        self.var = var
        self.kappa = large.field.degree.bit_length() - small.field.degree.bit_length()
        self.l_prime = self.var - self.kappa
        self.fri_binius = FRIBinius(self.large, self.l_prime, log_inv_rate)

    def _large_small_dot(self, log_length: int, large_vector: list[L], small_vector: list[K]) -> L:
        # abusing the implementation of towers. product will inherit the type of the first operand.
        # in general, we would need a K which supports a method `.upcast() -> L`.
        return sum(
            (large_vector[i] * small_vector[i] for i in range(1 << log_length)),  # type: ignore
            self.large.zero(),
        )

    def _combine(self, coordinates: list[K]) -> L:
        # in this and the below, i am "cheating" by assuming towers implicitly.
        return sum(
            (self.large(coordinates[v].value << v * self.small.field.degree) for v in range(1 << self.kappa)),
            self.large.zero(),
        )

    def _decompose(self, element: L) -> list[K]:
        mask = (1 << self.small.field.degree) - 1
        return [self.small(element.value >> v * self.small.field.degree & mask) for v in range(1 << self.kappa)]

    def commit(self, multilinear: list[K]) -> None:
        assert len(multilinear) == 1 << self.var
        packed_polynomial = [
            self._combine([multilinear[w << self.kappa | v] for v in range(1 << self.kappa)])
            for w in range(1 << self.l_prime)
        ]
        self.multilinear = multilinear  # only will need to use this once below; to compute ŝ
        self.fri_binius.commit(packed_polynomial)

    def s_hat(self, r: list[L]) -> TensorAlgElem[K, L]:
        self.r_high_tensor = tensor_expand(self.large, r[self.kappa :], self.l_prime)
        columns = [
            self._large_small_dot(
                self.l_prime,
                self.r_high_tensor,
                [self.multilinear[w << self.kappa | v] for w in range(1 << self.l_prime)],
            )
            for v in range(1 << self.kappa)
        ]
        return TensorAlgElem(self.small, self.large, columns)

    def receive_r_double_prime(self, r_double_prime: list[L]) -> None:
        r_double_prime_tensor = tensor_expand(self.large, r_double_prime, self.kappa)
        a = [
            self._large_small_dot(self.kappa, r_double_prime_tensor, self._decompose(self.r_high_tensor[w]))
            for w in range(1 << self.l_prime)
        ]
        composition = Polynomial(self.large, 2, {tuple([1, 1]): self.large.one()})
        self.sumcheck = Sumcheck(self.large, [self.fri_binius.multilinear, a], composition, False)

    def finalize(self) -> L:
        # prepares the underlying FRI-Binius object to roll.
        self.fri_binius.initialize_proof(self.sumcheck.challenges)  # self.sumcheck.challenges == r'
        # note: the above is wasteful; since internally it will tensor-expand `self.challenges`.
        # we more-or-less already did that, during our sumcheck---at least, before switchover, we started to.
        # in any case, the point of this model is illustrative---we have an artificial separation between the
        # ring-switching class and the underlying L-PCS (which, here, we elect to instantiate using a FRI-Binius).
        # of course in practice, we can, and will, integrate these---i.e., do Section 6, as opposed to Section 4.
        # so this wasted work won't be present in the real thing.
        return self.sumcheck.multilinears[0][0]  # should equal s' === t'(r')!!!
