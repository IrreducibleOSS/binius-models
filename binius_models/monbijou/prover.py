from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem
from ..finite_fields.utils import tensor_expand
from ..ips.polynomial import Polynomial
from ..ips.sumcheck import Sumcheck
from .constraint_system import ConstraintSystem, Key, Record, ShiftOp
from .utils import univariate_expand

F = TypeVar("F", bound=FiniteFieldElem)


MultilinearTriplet = dict[ShiftOp, list[F]]


class Prover(Generic[F]):
    def __init__(self, field: type[F], constraint_system: ConstraintSystem, record: Record, witness: list[int]):
        self.field = field
        self.constraint_system = constraint_system
        self.witness = witness
        assert len(witness) == 1 << constraint_system.l_words
        assert all(witness[i] in range(1 << 64) for i in range(1 << constraint_system.l_words))
        assert 1 << constraint_system.l_and == len(constraint_system.and_constraints)

        self.record = record

        # Batching challenge to be sampled in first phase
        self.lambda_val = field.zero()

        # Tensor of r_x_prime to be computed in first phase
        # and used in both phases
        self.r_x_prime_tensor: list[F] = []

    def _directly_evaluate(self, r_zhat_prime, r_x_prime):
        # directly computes and coughs up the values of â(Ẑ, X), b̂(Ẑ, X) and ĉ(Ẑ, X) on given inputs r'_ẑ, r'_x.
        # verifier can use the resulting things as the claims α_a, α_b, α_c to feed into the shift reduction.
        # in practice, these values will probably "pop out" for free at the end of the prover's univariate skip.
        # as for the verifier, my guess is that the prover will nondeterministically send them to the verifier,
        # so that the verifier can wrap up the univariate skip. pending the correctness of the supplied values.
        # the verifier will thus be done with univariate skip, and can turn his attention to the shift reduction.

        # this method can be viewed as a stand-in for the prover's univariate skip, in terms of the state we're left in.
        # will take this opportunity to memoize l_tilde ≔ L̃(r'_ẑ, i); the prover would compute this at the end of skip

        # Populate the a,b,c columns
        a = [0] * (1 << self.constraint_system.l_and)
        b = [0] * (1 << self.constraint_system.l_and)
        c = [0] * (1 << self.constraint_system.l_and)

        for x, (I_a, I_b, I_c) in enumerate(self.constraint_system.and_constraints):
            for y, op, s in I_a:
                a[x] ^= op(self.witness[y], s)
            for y, op, s in I_b:
                b[x] ^= op(self.witness[y], s)
            for y, op, s in I_c:
                c[x] ^= op(self.witness[y], s)
        # and check all rows satisfy the constraint
        assert all(a[i] & b[i] == c[i] for i in range(1 << self.constraint_system.l_and))

        assert len(r_x_prime) == self.constraint_system.l_and
        self.l_tilde = univariate_expand(self.field, r_zhat_prime)
        self.r_x_prime = r_x_prime

        def _extrapolate_bitstring(value):
            # interpreting `value` as the (bit-valued!) evaluations of some univariate polynomial over the domain D,
            # returns that polynomial's value at r'_ẑ. because we precomputed L̃(r'_ẑ, i), just now need a subset sum.
            return sum((self.l_tilde[i] for i in range(64) if value >> i & 1), self.field.zero())

        def _evaluate(m):
            univariate = [_extrapolate_bitstring(m[i]) for i in range(1 << self.constraint_system.l_and)]  # specialize
            for i in range(self.constraint_system.l_and):  # now fold
                for j in range(1 << self.constraint_system.l_and - i - 1):
                    univariate[j] = univariate[j << 1] + r_x_prime[i] * (univariate[j << 1 | 1] - univariate[j << 1])
            return univariate[0]

        return (_evaluate(a), _evaluate(b), _evaluate(c))

    # Sums the values in the tensor corresponding to the indices in x_indices
    def _accumulate_across_x(self, x_indices: list[int]) -> F:
        return sum((self.r_x_prime_tensor[x] for x in x_indices), self.field.zero())

    # Builds a multilinear triplet from a record.
    # This is the bottleneck of the prover.
    # In particular the j-loop is the bottleneck.
    def _build_triplets_from_record(
        self, record: Record
    ) -> tuple[MultilinearTriplet, MultilinearTriplet, MultilinearTriplet]:
        assert len(self.witness) == len(record)

        triplets = (
            {
                ShiftOp.SLL: [self.field.zero()] * (64 * 64),
                ShiftOp.SRL: [self.field.zero()] * (64 * 64),
                ShiftOp.SRA: [self.field.zero()] * (64 * 64),
            },
            {
                ShiftOp.SLL: [self.field.zero()] * (64 * 64),
                ShiftOp.SRL: [self.field.zero()] * (64 * 64),
                ShiftOp.SRA: [self.field.zero()] * (64 * 64),
            },
            {
                ShiftOp.SLL: [self.field.zero()] * (64 * 64),
                ShiftOp.SRL: [self.field.zero()] * (64 * 64),
                ShiftOp.SRA: [self.field.zero()] * (64 * 64),
            },
        )

        # Note the j-loop is now the inner-most.
        # Benchmarking for RSA the original loop
        # ordering is about 2.5 times slower.
        for original_word, lists in zip(self.witness, record):
            for m, keys in enumerate(lists):
                for key in keys:
                    word = original_word
                    multilinear = triplets[m][key.op]
                    acc = self._accumulate_across_x(key.x_indices)
                    for j in range(64):
                        if word & 1:
                            multilinear[key.amount << 6 | j] += acc
                        word >>= 1

        return triplets

    def initiate_first_phase(self, lambda_val):
        self.lambda_val = lambda_val

        # Compute the h multilinear triplet
        h: MultilinearTriplet = {
            ShiftOp.SLL: [self.field.zero()] * (64 * 64),
            ShiftOp.SRL: [self.field.zero()] * (64 * 64),
            ShiftOp.SRA: [self.field.zero()] * (64 * 64),
        }

        # h multilinears are indexed by s and j, s the more significant variable.
        for s in range(64):
            for j in range(64):
                h[ShiftOp.SLL][s << 6 | j] = self.l_tilde[j + s] if j + s < 64 else self.field.zero()
                val = self.l_tilde[j - s] if s <= j else self.field.zero()
                h[ShiftOp.SRL][s << 6 | j] = val
                h[ShiftOp.SRA][s << 6 | j] = val
        for s in range(1, 64):
            h[ShiftOp.SRA][s << 6 | 63] += h[ShiftOp.SRA][s - 1 << 6 | 63]

        # Compute the g multilinear triplet
        self.r_x_prime_tensor = tensor_expand(self.field, self.r_x_prime, self.constraint_system.l_and)

        # Compute one multilinear triplet for each of a, b, c
        triplets = self._build_triplets_from_record(self.record)

        # The g multilinear triplet has g.op the lambda_val-linear combination
        # of triplet_a.op, triplet_b.op, and triplet_c.op
        # these g's are actually surprisingly large. we have 9 × 64 × 64 field elements to store; this is ≈≈ 576 KiB.
        # this is about as much memory as we are liable to need in the second phase, for n_{words} == 15, so it seems ok
        # and moreover, we will do better than this _at best_ by a factor of 3, in view of how the sumcheck below goes.
        # optimizing this seems not simple, and not sure if we need to go down that path or not.
        g: MultilinearTriplet = {
            op: [
                sum(
                    (self.lambda_val**m * triplet[op][k] for m, triplet in enumerate(triplets)),
                    self.field.zero(),
                )
                for k in range(64 * 64)
            ]
            for op in [ShiftOp.SLL, ShiftOp.SRL, ShiftOp.SRA]
        }

        # Perform the sumcheck
        multilinears = [
            h[ShiftOp.SLL],
            g[ShiftOp.SLL],
            h[ShiftOp.SRL],
            g[ShiftOp.SRL],
            h[ShiftOp.SRA],
            g[ShiftOp.SRA],
        ]

        composition = Polynomial(  # ∑ₒₚ hₒₚ ⋅ gₒₚ
            self.field,
            6,
            {
                (1, 1, 0, 0, 0, 0): self.field.one(),
                (0, 0, 1, 1, 0, 0): self.field.one(),
                (0, 0, 0, 0, 1, 1): self.field.one(),
            },
        )
        self.first_phase_sumcheck = Sumcheck(self.field, multilinears, composition)

    def initiate_second_phase(self):
        # assert: self.first_phase_sumcheck.round == 12
        r_j = self.first_phase_sumcheck.challenges[:6]
        r_s = self.first_phase_sumcheck.challenges[6:]

        # Compute the partially folded witness
        r_j_tensor = tensor_expand(self.field, r_j, 6)
        w_r_j = [
            sum(
                (r_j_tensor[j] if self.witness[y] >> j & 1 else self.field.zero() for j in range(64)), self.field.zero()
            )
            for y in range(1 << self.constraint_system.l_words)
        ]

        # Compute the "monster" multilinear

        h_op_r_j_r_s = {
            ShiftOp.SLL: self.first_phase_sumcheck.multilinears[0][0],
            ShiftOp.SRL: self.first_phase_sumcheck.multilinears[2][0],
            ShiftOp.SRA: self.first_phase_sumcheck.multilinears[4][0],
        }

        r_s_tensor = tensor_expand(self.field, r_s, 6)

        # M can be computed flattened or not. I have yet to benchmark their relative speeds
        # But flattened will certainly be no slower for reasonable witness sizes

        # # UNFLATTENED
        # def _accumulate_keys(keys: list[Key]) -> F:
        #     return sum(
        #         (
        #             h_op_r_j_r_s[key.op] * r_s_tensor[key.amount] * self._accumulate_across_x(key.x_indices)
        #             for key in keys
        #         ),
        #         self.field.zero(),
        #     )

        # M = [
        #     sum((self.lambda_val**m * _accumulate_keys(keys) for m, keys in enumerate(lists)), self.field.zero())
        #     for lists in self.record
        # ]

        # FLATTENED
        scalars = [self.field.zero()] * (3 * 3 * 64)

        # Populate the flattened scalars.
        # Hopefully verbosity of indexing gives clarity
        for m, _ in enumerate("abc"):
            for op in ShiftOp:
                m_op_idx = m * 3 + list(ShiftOp).index(op)
                m_op_scalar = self.lambda_val**m * h_op_r_j_r_s[op]
                for s in range(64):
                    m_op_s_idx = m_op_idx << 6 | s
                    m_op_s_scalar = m_op_scalar * r_s_tensor[s]
                    scalars[m_op_s_idx] = m_op_s_scalar

        def _accumulate_for_m_keys(m: int, keys: list[Key]) -> F:
            return sum(
                (
                    scalars[m * 3 + list(ShiftOp).index(key.op) << 6 | key.amount]
                    * self._accumulate_across_x(key.x_indices)
                    for key in keys
                ),
                self.field.zero(),
            )

        M = [
            sum((_accumulate_for_m_keys(m, keys) for m, keys in enumerate(lists)), self.field.zero())
            for lists in self.record
        ]

        # Peform the sumcheck
        multilinears = [w_r_j, M]
        composition = Polynomial(self.field, 2, {(1, 1): self.field.one()})
        self.second_phase_sumcheck = Sumcheck(self.field, multilinears, composition)
