from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem
from .constraint_system import ConstraintSystem
from .prover import Prover

F = TypeVar("F", bound=FiniteFieldElem)


class Verifier(Generic[F]):
    def __init__(self, field: type[F], constraint_system: ConstraintSystem, prover: Prover[F]):
        self.field = field
        self.constraint_system = constraint_system
        self.prover = prover

    def verify_shift_reduction(self, alpha_a, alpha_b, alpha_c):
        (lambda_a, lambda_b, lambda_c) = (self.field.random(), self.field.random(), self.field.random())
        self.prover.initiate_first_phase(lambda_a, lambda_b, lambda_c)

        claim = lambda_a * alpha_a + lambda_b * alpha_b + lambda_c * alpha_c

        # begin first phase of sumcheck
        r_s = []
        for _ in range(6):
            evaluations = self.prover.first_phase_sumcheck.compute_round_polynomial()
            evaluations.insert(0, evaluations[0] + claim)
            challenge = self.field.random()
            self.prover.first_phase_sumcheck.receive_challenge(challenge)
            claim = self.prover.first_phase_sumcheck.interpolate(evaluations, challenge)
            r_s.append(challenge)
        r_j = []
        for _ in range(6):
            evaluations = self.prover.first_phase_sumcheck.compute_round_polynomial()
            evaluations.insert(0, evaluations[0] + claim)
            challenge = self.field.random()
            self.prover.first_phase_sumcheck.receive_challenge(challenge)
            claim = self.prover.first_phase_sumcheck.interpolate(evaluations, challenge)
            r_j.append(challenge)
