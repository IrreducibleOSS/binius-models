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
