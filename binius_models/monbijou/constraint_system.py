from dataclasses import dataclass
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


@dataclass
class Key:
    op: ShiftOp
    amount: int
    x_indices: list[int]


# A Record is constructed for each operand.
# A Record is a list with an item for each word.
# The item for a word specifies how the word is used as an operand.
# The item is a list of Keys.
# For every shift variant and shift amount applied to the word there is a Key in this list.
# Along with the shift variant and shift amount, the Key also stores the indices of
# the constraints in which that shifted word is used.
Record = list[list[Key]]
Operand = list[tuple[int, ShiftOp, int]]  # list of (y index, shift op, shift amount)
Constraint = tuple[Operand, Operand, Operand]


class ConstraintSystem:
    def __init__(self, l_and: int, l_words: int, and_constraints: list[Constraint]):
        self.l_and = l_and
        self.l_words = l_words
        self.and_constraints = and_constraints
        assert len(self.and_constraints) == 1 << self.l_and

    # Preprocess a record for each operand
    def _build_record(self, operands: tuple[Operand, ...]) -> Record:
        record: Record = [[] for _ in range(1 << self.l_words)]
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

    def build_records(self) -> tuple[Record, ...]:  # will be exactly 3 in tuple, but mypy doesn't like that
        return tuple(self._build_record(operands) for operands in zip(*self.and_constraints))
