from .constraint_system import ConstraintSystem
from .field import field
from .prover import Prover


class Verifier:
    def __init__(self, constraint_system: ConstraintSystem, prover: Prover):
        self.constraint_system = constraint_system
        self.prover = prover

    def verify_shift_reduction(self, alpha_a, alpha_b, alpha_c):
        (lambda_a, lambda_b, lambda_c) = (field.Random(), field.Random(), field.Random())
        self.prover.initiate_first_phase(lambda_a, lambda_b, lambda_c)

        claim = lambda_a * alpha_a + lambda_b * alpha_b + lambda_c * alpha_c

        # begin first phase of sumcheck
        r_s = []
        for _ in range(6):
            evaluations = self.prover.first_phase_sumcheck.compute_round_polynomial()
            evaluations.insert(0, evaluations[0] + claim)
            challenge = field.Random()
            self.prover.first_phase_sumcheck.receive_challenge(challenge)
            claim = self.prover.first_phase_sumcheck.interpolate(evaluations, challenge)
            r_s.append(challenge)
        r_j = []
        for _ in range(6):
            evaluations = self.prover.first_phase_sumcheck.compute_round_polynomial()
            evaluations.insert(0, evaluations[0] + claim)
            challenge = field.Random()
            self.prover.first_phase_sumcheck.receive_challenge(challenge)
            claim = self.prover.first_phase_sumcheck.interpolate(evaluations, challenge)
            r_j.append(challenge)
