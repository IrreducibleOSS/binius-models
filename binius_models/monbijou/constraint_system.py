from enum import Enum


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


class ConstraintSystem:
    def __init__(self, l_and: int, l_words: int, and_constraints):
        self.l_and = l_and
        self.l_words = l_words
        self.and_constraints = and_constraints
        assert len(self.and_constraints) == 1 << self.l_and
