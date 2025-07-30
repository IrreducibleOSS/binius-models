from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem
from ..finite_fields.utils import tensor_expand
from ..ips.polynomial import Polynomial
from ..ips.sumcheck import Sumcheck
from .constraint_system import ConstraintSystem, Key, ShiftOp
from .utils import univariate_expand

F = TypeVar("F", bound=FiniteFieldElem)


MultilinearTriplet = dict[ShiftOp, list[F]]


class Prover(Generic[F]):
    def __init__(self, field: type[F], constraint_system: ConstraintSystem, keys: list[Key], witness: list[int]):
        self.field = field
        self.constraint_system = constraint_system
        self.witness = witness
        assert len(witness) == 1 << constraint_system.l_words
        assert all(witness[i] in range(1 << 64) for i in range(1 << constraint_system.l_words))
        assert 1 << constraint_system.l_and == len(constraint_system.and_constraints)

        # The prover will use these keys as a representation
        # of the constraint system when proving.
        self.keys = keys

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

    def build_g_triplet(self) -> MultilinearTriplet:
        # Allocate the triplet
        triplets = [{op: [self.field.zero()] * (64 * 64) for op in ShiftOp} for _ in range(3)]

        # Traverse the keys (not the words!),
        # identifying the multilinear into
        # which to accumulate and the word
        # to accumulate into it.
        for key in self.keys:
            multilinear = triplets[key.m][key.op]
            val = key.accumulate(self.field, self.r_x_prime_tensor)
            word = self.witness[key.y]
            for j in range(64):
                if word & 1:
                    multilinear[key.s << 6 | j] += val
                word >>= 1

        # Combine the multilinears across the three operands using lambda
        return {
            op: [
                sum(
                    (self.lambda_val**m * triplet[op][k] for m, triplet in enumerate(triplets)),
                    self.field.zero(),
                )
                for k in range(64 * 64)
            ]
            for op in [ShiftOp.SLL, ShiftOp.SRL, ShiftOp.SRA]
        }

    def initiate_first_phase(self, lambda_val):
        self.lambda_val = lambda_val

        # This tensor expansion is needed for both phases
        self.r_x_prime_tensor = tensor_expand(self.field, self.r_x_prime, self.constraint_system.l_and)

        # Compute the h multilinear triplet
        h: MultilinearTriplet = {
            ShiftOp.SLL: [self.field.zero()] * (64 * 64),
            ShiftOp.SRL: [self.field.zero()] * (64 * 64),
            ShiftOp.SRA: [self.field.zero()] * (64 * 64),
        }

        # h multilinears are indexed by s and j, s the more significant variable.
        for s in range(64):
            for j in range(64 - s):
                h[ShiftOp.SLL][s << 6 | j] = self.l_tilde[j + s]
            for j in range(s, 64):
                h[ShiftOp.SRL][s << 6 | j] = self.l_tilde[j - s]
                h[ShiftOp.SRA][s << 6 | j] = self.l_tilde[j - s]
        for s in range(1, 64):
            h[ShiftOp.SRA][s << 6 | 63] += h[ShiftOp.SRA][s - 1 << 6 | 63]

        # Compute the g multilinear triplet
        g = self.build_g_triplet()

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

        # Allocate M
        M = [self.field.zero()] * (1 << self.constraint_system.l_words)

        # Traverse the keys (not the words)
        # and identify the index of M into
        # which to accumulate.
        for key in self.keys:
            base = key.accumulate(self.field, self.r_x_prime_tensor)
            m_op_idx = key.m * 3 + list(ShiftOp).index(key.op)
            m_op_s_idx = m_op_idx << 6 | key.s
            M[key.y] += base * scalars[m_op_s_idx]

        # Peform the sumcheck
        multilinears = [w_r_j, M]
        composition = Polynomial(self.field, 2, {(1, 1): self.field.one()})
        self.second_phase_sumcheck = Sumcheck(self.field, multilinears, composition)
