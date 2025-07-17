import random

from .constraint_system import ConstraintSystem, ShiftOp
from .field import field
from .prover import Prover, Witness
from .verifier import Verifier

constraint_system = ConstraintSystem(
    l_and=3,
    l_words=6,
    and_constraints=[
        (  # (w[2] ~>> 5 ^ w[5] >> 5) & w[10] << 55 == w[63] ---- just a random constraint that I can mock
            [(2, ShiftOp.SRA, 5), (5, ShiftOp.SRL, 5)],
            [(10, ShiftOp.SLL, 55)],
            [(63, ShiftOp.SLL, 0)],
        ),
        (  # trivial constraint, since it insists that SOMETHING ^ SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING ^ SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING ^ SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING ^ SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING ^ SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING ^ SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING ^ SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
    ],
)


def test_good():
    data = [random.randrange(1 << 64) for _ in range(1 << constraint_system.l_words)]
    data[63] = (ShiftOp.SRA(data[2], 5) ^ ShiftOp.SRL(data[5], 5)) & ShiftOp.SLL(data[10], 55)
    witness = Witness(data)
    prover = Prover(constraint_system, witness)

    r_zhat_prime = field.Random()
    r_x_prime = [field.Random() for _ in range(constraint_system.l_and)]
    (alpha_a, alpha_b, alpha_c) = prover._directly_evaluate(r_zhat_prime, r_x_prime)

    verifier = Verifier(constraint_system)
    verifier.verify_shift_reduction(prover, alpha_a, alpha_b, alpha_c)


# negative tests, to do:
# test a thing where the witness isn't valid, but α_c, α_b, α_c are valid.
# test a thing where the witness is valid, but α_c, α_b, α_c aren't.
