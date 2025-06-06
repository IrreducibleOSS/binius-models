from math import ceil, log2
from typing import TypeVar

from ..finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField
from ..utils.utils import is_bit_set, is_power_of_two
from .additive_ntt import GaoMateerBasis

F = TypeVar("F", bound=BinaryTowerFieldElem)


class Elem1bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(0)


class Elem2bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(1)


class Elem4bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(2)


class Elem8bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(3)


class Elem16bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(4)


class Elem32bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(5)


class Elem64bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(6)


class Elem128bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


levels = [Elem1bFP, Elem2bFP, Elem4bFP, Elem8bFP, Elem16bFP, Elem32bFP, Elem64bFP, Elem128bFP]


class FrobeniusNTT:
    # first-pass implementation that assumes bit-level access granularity;
    # todo in a further pass: bitslicing
    def __init__(self, log_h: int, log_inv_rate: int) -> None:
        self.log_h = log_h
        self.log_inv_rate = log_inv_rate

        initial_dimension = self.log_h + self.log_inv_rate
        indeterminates_needed = ceil(log2(initial_dimension))

        mateer = GaoMateerBasis(Elem128bFP, log_h, 2)
        self.basis = mateer.constants[0]
        for i in range(indeterminates_needed):  # revisit whether to keep this around
            for j in range(1 << i):
                self.basis[1 << i | j] = self.basis[1 << i | j].downcast(levels[i + 1])

    def _encode_helper(
        self,
        input: list[Elem1bFP],
        output: list[Elem1bFP],
        l: int,
        coefficient_level: int,
        alpha_idx: int,
        alpha_length: int,
        alpha_level: int,
        beta_idx: int,
    ) -> None:
        # todo: make this sensitive to the fact that all but first 1 << log_h elements of input are zero
        # it should be possible to infer alpha_length as just self.log_h + self.log_inv_rate - l
        # alpha represents the index of this block within the current butterfly phase.
        # keeping track of what field it's defined over is tricky and important.
        # invariant: 1 << (1 << alpha_level) > alpha << 1 will hold (i.e., alpha << 1 is defined over T_{alpha_level}).
        # also invariant: alpha_idx is alpha_length bits long.
        # alpha_length is going to be the bit-length of alpha, EXCLUDING leading zeros---the length of its "content".
        # alpha_level is going to be the tower level that twiddle = MateerBasis(alpha << 1) is defined over.
        # recall that to compute the twiddle, we need to skip the least-significant basis vector, namely 1.
        # if alpha = 0, then recursive sub-lengths will be 0 and 1; sub-levels will be 0 and 1.
        # otherwise, recursive sub-lengths will both be alpha_length + 1. moreover, recursive sub-levels will both be
        # 1 << alpha_level >= alpha_length + 2 ? alpha_level : alpha_level + 1.
        # this is equivalent to guaranteeing that in the NEXT recursive call the invariant will hold.

        if l == 0:
            # coefficient level is the field we're in.
            # alpha_length is actually the Sigma we're in (!?).
            return

        special = alpha_length != 0 and is_power_of_two(alpha_length)
        twiddle = sum(
            (self.basis[i + 1] for i in range(alpha_length) if is_bit_set(alpha_idx, i)), levels[alpha_level].zero()
        )

        # todo below: smartly handle the case where l > self.log_h, and the lower half of the input is zero.
        input_stash = [input[i] + twiddle * input[1 << l - 1 | i] for i in range(1 << l - 1)]
        if alpha_level > coefficient_level:
            input_stash = [element.upcast(levels[alpha_level]) for element in input_stash]
        self._encode_helper(
            input_stash,
            output,
            l - 1,
            alpha_level,
            alpha_idx << 1,
            0 if alpha_idx == 0 else alpha_length + 1,
            alpha_level if alpha_idx == 0 or 1 << alpha_level >= alpha_length + 2 else alpha_level + 1,
            beta_idx if special else beta_idx << 1,
        )

        if special:
            return
        input_stash = [input_stash[i] + input[1 << l - 1 | i] for i in range(1 << l - 1)]
        # these will inherit the first operand's field, which is at least as large as the second; so no need to upcast.
        self._encode_helper(
            input_stash,
            output,
            l - 1,
            alpha_level,
            alpha_idx << 1 | 1,
            alpha_length + 1,
            alpha_level if 1 << alpha_level >= alpha_length + 2 else alpha_level + 1,
            beta_idx << 1 | 1,
        )

    def encode(self, input: list[Elem1bFP]) -> list[Elem1bFP]:
        assert len(input) == 1 << self.log_h
        input = input + [Elem1bFP.zero()] * (((1 << self.log_inv_rate) - 1) << self.log_h)
        output = [Elem1bFP.zero()] * (1 << self.log_h + self.log_inv_rate)
        self._encode_helper(input, output, self.log_h + self.log_inv_rate, 0, 0, 0, 0, 0)
        return output
