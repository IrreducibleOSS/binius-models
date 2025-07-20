from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem
from .constraint_system import ConstraintSystem, ShiftOp
from .prover import Prover
from .utils import tensor_expand, univariate_expand

F = TypeVar("F", bound=FiniteFieldElem)


class Verifier(Generic[F]):
    def __init__(self, field: type[F], constraint_system: ConstraintSystem, prover: Prover[F]):
        self.field = field
        self.constraint_system = constraint_system
        self.prover = prover

    def _evaluate_h_op(self, r_zhat_prime: F, r_j: list[F], r_s: list[F]) -> F:
        # to figure out. whether and how we can gain by returning all 3 of these at once.
        l_tilde = univariate_expand(self.field, r_zhat_prime)
        # now comes the hard part!
        s = [self.field.one()] * 64
        s_prime = [self.field.zero()] * 64
        a = [self.field.zero()] * 64
        j_product = self.field.one
        for k in range(6):
            # precompute some common subexpressions that we are going to re-use.
            r_j_r_s = r_j[k] * r_s[k]  # jₖ ⋅ sₖ
            r_j_one_rs = r_j[k] - r_j_r_s  # jₖ ⋅ (1 - sₖ)
            one_r_j_rs = r_s[k] - r_j_r_s  # (1 - jₖ) ⋅ sₖ
            thing = self.field.one() + r_j[k] + r_s[k]  # 1 + jₖ + sₖ
            for i in range(1 << k):
                s[1 << k | i] = r_j_one_rs * s[i]  # write upper halves first
                s_prime[1 << k | i] = one_r_j_rs * s[i] + thing * s_prime[i]  # Iₖ = 1

                s[i] = thing * s[i] + r_j_one_rs * s_prime[i]
                s_prime[i] *= one_r_j_rs

                a[1 << k | i] = r_s[k] + (self.field.one() + r_s[k]) * a[i]
                a[i] += a[1 << k | i] - r_s[k]  # this is a bit of a trick; wastes an add, but saves a write to a temp
            j_product *= r_j[k]
        return {
            ShiftOp.SLL: s,
            ShiftOp.SRL: s,  # ????????? how will i handle the transmutation of indices?
            ShiftOp.SRA: [s[i] + j_product * a[i] for i in range(64)],
        }

    def verify_shift_reduction(self):
        r_zhat_prime = self.field.random()
        r_x_prime = [self.field.random() for _ in range(self.constraint_system.l_and)]

        # ask for the below quantities directly from the prover.
        # in real life, these will be the final outputs of univariate skip reduction, and will come from that.
        (alpha_a, alpha_b, alpha_c) = self.prover._directly_evaluate(r_zhat_prime, r_x_prime)

        # begin actual verification of shift reduction
        (lambda_a, lambda_b, lambda_c) = (self.field.random(), self.field.random(), self.field.random())
        claim = lambda_a * alpha_a + lambda_b * alpha_b + lambda_c * alpha_c

        self.prover.initiate_first_phase(lambda_a, lambda_b, lambda_c)
        for _ in range(12):
            evaluations = self.prover.first_phase_sumcheck.compute_round_polynomial()
            evaluations.insert(0, evaluations[0] + claim)
            challenge = self.field.random()
            self.prover.first_phase_sumcheck.receive_challenge(challenge)
            claim = self.prover.first_phase_sumcheck.interpolate(evaluations, challenge)
        r_s, r_j = self.prover.first_phase_sumcheck.challenges[:6], self.prover.first_phase_sumcheck.challenges[6:]

        self.prover.initiate_second_phase()  # don't need to send r_s and r_j; prover memoized these during sumcheck
        for _ in range(self.constraint_system.l_words):
            evaluations = self.prover.second_phase_sumcheck.compute_round_polynomial()
            evaluations.insert(0, evaluations[0] + claim)
            challenge = self.field.random()
            self.prover.second_phase_sumcheck.receive_challenge(challenge)
            claim = self.prover.second_phase_sumcheck.interpolate(evaluations, challenge)
        r_y = self.prover.second_phase_sumcheck.challenges[:]

        # begin computation of final evaluations.
        lambdas = {
            "a": lambda_a,
            "b": lambda_b,
            "c": lambda_c,
        }
        r_s_tensor = tensor_expand(self.field, r_s, 6)  # prover already had to compute this
        r_j_tensor = tensor_expand(self.field, r_j, 6)  # prover already had to compute this
        r_x_prime_tensor = tensor_expand(self.field, r_x_prime, self.constraint_system.l_and)
        r_y_tensor = tensor_expand(self.field, r_y, self.constraint_system.l_words)

        # NOTE: compute w̃(r_j, r_y) manually for our (testing) purposes; in real life it will be delegated to the PCS
        w_r_j_r_y = sum(
            (
                sum((r_j_tensor[j] for j in range(64) if self.prover.witness[y] >> j & 1), self.field.zero())
                * r_y_tensor[y]
                for y in range(1 << self.constraint_system.l_words)
            ),
            self.field.zero(),
        )

        right_hand_multilinear = self.field.zero()
        M = {  # will contain M̃ₘ,ₒₚ(r'_x, r_y, r_s)
            ShiftOp.SLL: {
                "a": self.field.zero(),
                "b": self.field.zero(),
                "c": self.field.zero(),
            },
            ShiftOp.SRL: {
                "a": self.field.zero(),
                "b": self.field.zero(),
                "c": self.field.zero(),
            },
            ShiftOp.SRA: {
                "a": self.field.zero(),
                "b": self.field.zero(),
                "c": self.field.zero(),
            },
        }
        for i, (I_a, I_b, I_c) in enumerate(self.constraint_system.and_constraints):
            for j, op, s in I_a:
                M[op]["a"] += r_x_prime_tensor[i] * r_y_tensor[j] * r_s_tensor[s]
            for j, op, s in I_b:
                M[op]["b"] += r_x_prime_tensor[i] * r_y_tensor[j] * r_s_tensor[s]
            for j, op, s in I_c:
                M[op]["c"] += r_x_prime_tensor[i] * r_y_tensor[j] * r_s_tensor[s]

        h_op = self._evaluate_h_op(r_zhat_prime, r_j, r_s)
        for op in [ShiftOp.SLL, ShiftOp.SRL, ShiftOp.SRA]:
            result = self.field.zero()
            for m, lambda_m in lambdas.items():
                result += lambda_m * M[op][m]
            result *= h_op[op]
            right_hand_multilinear += result

        assert claim == w_r_j_r_y * right_hand_multilinear  # just one check!
