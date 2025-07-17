from enum import Enum


class ShiftOp(Enum):
    def SLL(v, s):
        return v << s

    def SRL(v, s):
        return v >> s

    def SRA(v, s):
        return (v if v < 1 << 63 else v - (1 << 64)) >> s & (1 << 64) - 1

    def __call__(self, v, s):
        return self.value(v, s)


class ConstraintSystem:
    def __init__(
        self,
        l_and: int,
        l_words: int,
        and_constraints: list[
            tuple[list[tuple[int, ShiftOp, int]], list[tuple[int, ShiftOp, int]], list[tuple[int, ShiftOp, int]]]
        ],
    ):
        self.l_and = l_and
        self.l_words = l_words
        self.and_constraints = and_constraints
        assert len(self.and_constraints) == 1 << self.l_and
