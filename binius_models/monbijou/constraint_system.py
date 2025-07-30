from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from ..finite_fields.finite_field import FiniteFieldElem

F = TypeVar("F", bound=FiniteFieldElem)


class ShiftOp(Enum):
    SLL = "SLL"
    SRL = "SRL"
    SRA = "SRA"

    def __call__(self, v, s):
        if self == ShiftOp.SLL:
            return v << s & (1 << 64) - 1
        elif self == ShiftOp.SRL:
            return v >> s
        elif self == ShiftOp.SRA:
            return (v if v < 1 << 63 else v - (1 << 64)) >> s & (1 << 64) - 1
        else:
            raise ValueError(f"Unknown shift operation: {self}")


Operand = list[tuple[int, ShiftOp, int]]  # list of (y index, shift op, shift amount)
Constraint = tuple[Operand, Operand, Operand]


# A Key consists of (shift op, shift amount, operand index, word index, constraint indices)
# The word at index `y` in the witness, shifted `s` bits with shift variant `op`, participates
# as operand of index `m` in the constraints with indices in `xs`
@dataclass
class Key:
    op: ShiftOp
    s: int
    m: int
    y: int
    xs: list[int]

    def accumulate(self, field_type: type[F], tensor: list[F]) -> F:
        return sum((tensor[x] for x in self.xs), field_type.zero())


class ConstraintSystem:
    def __init__(self, l_and: int, l_words: int, and_constraints: list[Constraint]):
        self.l_and = l_and
        self.l_words = l_words
        self.and_constraints = and_constraints
        assert len(self.and_constraints) == 1 << self.l_and

    # For every constraint system, we build a list of keys
    def build_keys(self) -> list[Key]:
        # Build a list of keys for every word
        # We begin nested rather than flattened such that
        # all keys belongs to a word are grouped together
        nested_keys: list[list[Key]] = [[] for _ in range(1 << self.l_words)]

        for x, constraint in enumerate(self.and_constraints):
            for m, I_m in enumerate(constraint):
                for y, op, s in I_m:
                    keys = nested_keys[y]
                    existing_key = next((key for key in keys if key.op == op and key.s == s and key.m == m), None)
                    if existing_key:
                        existing_key.xs.append(x)
                    else:
                        keys.append(Key(op=op, s=s, m=m, y=y, xs=[x]))

        # Flatten and retain only the inner key lists that are non-empty
        return sum([keys for keys in nested_keys if keys], [])
