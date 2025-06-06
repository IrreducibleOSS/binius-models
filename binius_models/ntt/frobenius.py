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

        mateer = GaoMateerBasis(levels[indeterminates_needed], log_h, 2)
        self.basis = mateer.constants[0]
        self.basis[0].downcast(levels[0])
        for iota in range(indeterminates_needed):  # revisit whether to keep this around
            for i in range(1 << iota):
                self.basis[1 << iota | i] = self.basis[1 << iota | i].downcast(levels[iota + 1])

    def _lexicographic_to_field(self, index: int, length: int, iota: int) -> BinaryTowerFieldElem:
        # could do `range(i)` where "i" is as below.
        # we are summing over more than we need; index will only be i ≤ 1 << iota bits.
        return sum((self.basis[k] for k in range(length) if is_bit_set(index, k)), levels[iota].zero())

    def _field_to_lexicographic(self, field_index: int, iota: int) -> int:
        # TODO: this is wrong; we are off by reversing the isomorphism back into lexicographic coords.
        return field_index.value

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
            # alpha_length = m is actually the Sigma we're in (!?).
            # beta_idx is the index of the result within Σₘ.
            offset = 0 if alpha_level == 0 else 1 << alpha_length - 1  # offset == ⌊ log₂ alpha_index ⌋
            for i in range(1 << coefficient_level):
                output[offset | beta_idx << coefficient_level | i] = Elem1bFP(input[0].value >> i & 0x01)
            return

        special = is_power_of_two(alpha_length)
        twiddle = self._lexicographic_to_field(alpha_idx + 1, alpha_length << 1, alpha_level)

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
            beta_idx if alpha_idx == 0 or special else beta_idx << 1,
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
            beta_idx if alpha_idx == 0 else beta_idx << 1 | 1,
        )

    def encode(self, input: list[Elem1bFP]) -> list[Elem1bFP]:
        assert len(input) == 1 << self.log_h
        input = input + [Elem1bFP.zero()] * (((1 << self.log_inv_rate) - 1) << self.log_h)
        output = [Elem1bFP.zero()] * (1 << self.log_h + self.log_inv_rate)
        self._encode_helper(input, output, self.log_h + self.log_inv_rate, 0, 0, 0, 0, 0)
        return output

    def unpack_output(self, output: list[Elem1bFP]) -> Elem128bFP:  # <---- bigger than necessary
        # `output` is the condensed, raw bit-output of the Frobenius NTT.
        # `index`: an log_h + log_inv_rate-bit integer, index of the desired output element we want to conjure.
        unpacked = [Elem1bFP.zero()] * (1 << self.log_h + self.log_inv_rate)

        unpacked[0] = output[0]  # kill the 0 case right away, which is degenerate
        for i in range(1, self.log_h + self.log_inv_rate + 1):
            iota = ceil(log2(i))
            beta_bits = i - 1 - iota
            for j in range(1 << beta_bits):
                j_copy = j
                index = 1
                for k in range(1, i):
                    index <<= 1
                    if is_power_of_two(k):
                        continue
                    else:
                        index |= j_copy & 1  # <--- TODO: I think this is off by a bit-reversal of j_copy; revisit
                        j_copy >>= 1
                value = levels[iota](sum(output[1 << i - 1 | j << iota | k].value << k for k in range(1 << iota)))
                for _ in range(1 << iota):  # galois loop
                    unpacked[index] = value

                    field_index = self._lexicographic_to_field(index, i, iota)
                    field_index = field_index.square()  # overwrite
                    index = self._field_to_lexicographic(field_index, iota)
        return unpacked
