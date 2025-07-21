import random

import pytest

from ..finite_fields.ghash import GHASHElem
from .constraint_system import ConstraintSystem, ShiftOp
from .prover import Prover
from .verifier import Verifier

constraint_system = ConstraintSystem(  # a mock constraint system, hardcoded for now
    l_and=3,
    l_words=6,
    and_constraints=[
        (  # (w[2] ~>> 5 ^ w[5] >> 5) & w[10] << 55 == w[63]---just a random constraint that I can make hold manually
            [(2, ShiftOp.SRA, 5), (5, ShiftOp.SRL, 5)],
            [(10, ShiftOp.SLL, 55)],
            [(63, ShiftOp.SLL, 0)],
        ),
        (  # trivial constraint, since it insists that SOMETHING & SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING & SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING & SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING & SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING & SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING & SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
        (  # trivial constraint, since it insists that SOMETHING & SOMETHING = SOMETHING. this imposes no real condition
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
            [(7, ShiftOp.SLL, 10), (14, ShiftOp.SRL, 20), (21, ShiftOp.SRA, 30), (28, ShiftOp.SLL, 40)],
        ),
    ],
)


def test_good():
    witness = [random.randrange(1 << 64) for _ in range(1 << constraint_system.l_words)]
    witness[63] = (ShiftOp.SRA(witness[2], 5) ^ ShiftOp.SRL(witness[5], 5)) & ShiftOp.SLL(witness[10], 55)
    prover = Prover(GHASHElem, constraint_system, witness)

    # the below block of code "simulates" the univariate skip.
    r_zhat_prime = GHASHElem.random()
    r_x_prime = [GHASHElem.random() for _ in range(constraint_system.l_and)]
    alpha_a, alpha_b, alpha_c = prover._directly_evaluate(r_zhat_prime, r_x_prime)  # grab directly from prover
    # some remarks. right now, we are NOT actually checking whether the witness satisfies the constraint system!
    # actually, that will be the job of the univariate skip. at the end of that, the verifier will be assured that
    # f(Ẑ, X) vanishes identically on D × B_{ℓ_{and}}---and so satisfies the constraint system---PENDING correctness
    # of α_a, α_b and α_c, which claims will be nondeterministically supplied by the prover to the verifier.
    # thus the job of the skip reduction is NOT really to check anything about constraint satisfaction, but rather to
    # check whether claimed evaluations α_a, α_b and α_c of â(Ẑ, X), b̂(Ẑ, X), and ĉ(Ẑ, X) at r'_ẑ, r'_x are good.

    verifier = Verifier(GHASHElem, constraint_system, prover)
    verifier.verify_shift_reduction(r_zhat_prime, r_x_prime, alpha_a, alpha_b, alpha_c)


def test_bad():
    witness = [random.randrange(1 << 64) for _ in range(1 << constraint_system.l_words)]
    witness[63] = (ShiftOp.SRA(witness[2], 5) ^ ShiftOp.SRL(witness[5], 5)) & ShiftOp.SLL(witness[10], 55)
    prover = Prover(GHASHElem, constraint_system, witness)

    r_zhat_prime = GHASHElem.random()
    r_x_prime = [GHASHElem.random() for _ in range(constraint_system.l_and)]
    alpha_a, alpha_b, alpha_c = prover._directly_evaluate(r_zhat_prime, r_x_prime)  # grab directly from prover
    alpha_a += GHASHElem.one()  # corrupt one claim...

    verifier = Verifier(GHASHElem, constraint_system, prover)
    with pytest.raises(AssertionError):
        verifier.verify_shift_reduction(r_zhat_prime, r_x_prime, alpha_a, alpha_b, alpha_c)
