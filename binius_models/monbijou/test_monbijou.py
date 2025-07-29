import random

import pytest

from ..finite_fields.ghash import GHASHElem
from .constraint_system import ConstraintSystem, Operand, ShiftOp
from .prover import Key, Prover, Record
from .verifier import Verifier

# this is a silly, dummy constraint system, that basically asserts that (⋀ᵢ₌₀⁷ w[i] << i) >> 5 == w[15] ~>> 2.
# it puts the first 8 words into a tree of ANDs, shifting each thing by the appropriate amount.
# the point is to rig up something that is not completely trivial, but which I can hand-enforce.
# note how each list is of length 1. we are only including one thing (and not an XOR of things) in each AND operand.
# you can definitely modify this if you want, so that the lists are of length more than one.
constraint_system = ConstraintSystem(
    l_and=3,  # 8 constraints
    l_words=4,  # witness length is 16
    and_constraints=[
        (  # w[0] << 0 & w[1] << 1 == w[8]
            [(0, ShiftOp.SLL, 0)],
            [(1, ShiftOp.SLL, 1)],
            [(8, ShiftOp.SLL, 0)],
        ),
        (  # w[2] << 2 & w[3] << 3 == w[9]
            [(2, ShiftOp.SLL, 2)],
            [(3, ShiftOp.SLL, 3)],
            [(9, ShiftOp.SLL, 0)],
        ),
        (  # w[4] << 4 & w[5] << 5 == w[10]
            [(4, ShiftOp.SLL, 4)],
            [(5, ShiftOp.SLL, 5)],
            [(10, ShiftOp.SLL, 0)],
        ),
        (  # w[6] << 6 & w[7] << 7 == w[11]
            [(6, ShiftOp.SLL, 6)],
            [(7, ShiftOp.SLL, 7)],
            [(11, ShiftOp.SLL, 0)],
        ),
        (  # w[8] << 0 & w[9] << 0 == w[12]
            [(8, ShiftOp.SLL, 0)],
            [(9, ShiftOp.SLL, 0)],
            [(12, ShiftOp.SLL, 0)],
        ),
        (  # w[10] << 0 & w[11] << 0 == w[13]
            [(10, ShiftOp.SLL, 0)],
            [(11, ShiftOp.SLL, 0)],
            [(13, ShiftOp.SLL, 0)],
        ),
        (  # w[12] << 0 & w[13] << 0 == w[14]
            [(12, ShiftOp.SLL, 0)],
            [(13, ShiftOp.SLL, 0)],
            [(14, ShiftOp.SLL, 0)],
        ),
        (  # w[14] >> 5 & w[14] >> 5 == w[15] ~>> 2
            [(14, ShiftOp.SRL, 5)],
            [(14, ShiftOp.SRL, 5)],
            [(15, ShiftOp.SRA, 2)],
        ),
    ],
)

mask = (1 << 64) - 1
witness = [random.randrange(1 << 64) for _ in range(1 << constraint_system.l_words)]
witness[8] = witness[0] << 0 & witness[1] << 1 & mask
witness[9] = witness[2] << 2 & witness[3] << 3 & mask
witness[10] = witness[4] << 4 & witness[5] << 5 & mask
witness[11] = witness[6] << 6 & witness[7] << 7 & mask
witness[12] = witness[8] & witness[9]
witness[13] = witness[10] & witness[11]
witness[14] = witness[12] & witness[13]
witness[15] = witness[14] >> 3


# Preprocess a record for each operand
def build_record(n_words: int, operands: tuple[Operand, ...]) -> Record:
    record: Record = [[] for _ in range(n_words)]
    for x, operand in enumerate(operands):
        for y, op, s in operand:
            keys = record[y]
            # Find existing key with same op and s, or None if not found
            existing_key = next((key for key in keys if key.op == op and key.amount == s), None)
            if existing_key:
                existing_key.x_indices.append(x)
            else:
                keys.append(Key(op=op, amount=s, x_indices=[x]))
    return record


def test_good():
    n_words = 1 << constraint_system.l_words
    # Unzip the constraints into lists of operands
    a_operands, b_operands, c_operands = zip(*constraint_system.and_constraints)
    # Build a record for each list of operands
    a_record = build_record(n_words, a_operands)
    b_record = build_record(n_words, b_operands)
    c_record = build_record(n_words, c_operands)
    prover = Prover(GHASHElem, constraint_system, witness, a_record, b_record, c_record)

    # the below block of code "simulates" the univariate skip.
    r_zhat_prime = GHASHElem.random()
    r_x_prime = [GHASHElem.random() for _ in range(constraint_system.l_and)]
    alpha_a, alpha_b, alpha_c = prover._directly_evaluate(r_zhat_prime, r_x_prime)  # grab directly from prover
    # some remarks. right now, we are NOT actually checking whether the witness satisfies the constraint system!
    # actually, that will be the job of the univariate skip. at the end of that, the verifier will be assured that
    # f(Ẑ, X) vanishes identically on D × B_{ℓ_{and}}---and so satisfies the constraint system---PENDING correctness
    # of α_a, α_b and α_c, claims which will be nondeterministically supplied by the prover at the end of the skip.
    # thus the job of the shift reduction is NOT really to check anything about constraint satisfaction, but rather to
    # check whether claimed evaluations α_a, α_b and α_c of â(Ẑ, X), b̂(Ẑ, X), and ĉ(Ẑ, X) at r'_ẑ, r'_x are good.

    verifier = Verifier(GHASHElem, constraint_system, prover)
    verifier.verify_shift_reduction(r_zhat_prime, r_x_prime, alpha_a, alpha_b, alpha_c)


def test_bad():
    n_words = 1 << constraint_system.l_words
    # Unzip the constraints into lists of operands
    a_operands, b_operands, c_operands = zip(*constraint_system.and_constraints)
    # Build a record for each list of operands
    a_record = build_record(n_words, a_operands)
    b_record = build_record(n_words, b_operands)
    c_record = build_record(n_words, c_operands)
    prover = Prover(GHASHElem, constraint_system, witness, a_record, b_record, c_record)

    r_zhat_prime = GHASHElem.random()
    r_x_prime = [GHASHElem.random() for _ in range(constraint_system.l_and)]
    alpha_a, alpha_b, alpha_c = prover._directly_evaluate(r_zhat_prime, r_x_prime)  # grab directly from prover
    alpha_a += GHASHElem.one()  # corrupt one claim...

    verifier = Verifier(GHASHElem, constraint_system, prover)
    with pytest.raises(AssertionError):
        verifier.verify_shift_reduction(r_zhat_prime, r_x_prime, alpha_a, alpha_b, alpha_c)
