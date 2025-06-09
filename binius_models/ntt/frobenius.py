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
        self.basis[0] = self.basis[0].downcast(levels[0])
        for iota in range(indeterminates_needed):  # revisit whether to keep this around
            for i in range(1 << iota):
                self.basis[1 << iota | i] = self.basis[1 << iota | i].downcast(levels[iota + 1])

    def _square_in_coordinates(self, index: int) -> int:
        return index ^ index >> 1

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
            offset = 0 if alpha_length == 0 else 1 << alpha_length - 1  # offset == ⌊ log₂ alpha_index ⌋
            for i in range(1 << coefficient_level):
                output[offset | beta_idx << coefficient_level | i] = Elem1bFP(input[0].value >> i & 0x01)
            return

        special = is_power_of_two(alpha_length)
        twiddle = sum(
            (self.basis[k + 1] for k in range(alpha_length) if is_bit_set(alpha_idx, k)), levels[alpha_level].zero()
        )

        early_round = l > self.log_h
        folded = [input[i] if early_round else input[i] + twiddle * input[1 << l - 1 | i] for i in range(1 << l - 1)]
        # the python model is a bit quirky on this; it will allow us to multiply big * small and also add small + big.
        # the output of each binop will inherit the field of the first operand. thus it'll be `coefficient_field`.
        # of course this will be wrong whenever `alpha_level > coefficient_level`. thus we perform the upcast below.
        # i have delayed this upcast just for illustration---so that you can see that these are small × larges.
        # in general, we have `alpha_level` ≥ `coefficient_level`; the below will be a no-op except when this is strict.
        folded = [element.upcast(levels[alpha_level]) for element in folded]

        self._encode_helper(
            folded,
            output,
            l - 1,
            alpha_level,
            alpha_idx << 1,
            0 if alpha_length == 0 else alpha_length + 1,
            alpha_level if alpha_length == 0 or 1 << alpha_level >= alpha_length + 2 else alpha_level + 1,
            beta_idx if alpha_length == 0 or special else beta_idx << 1,
        )

        if special:
            return
        folded = [folded[i] if early_round else folded[i] + input[1 << l - 1 | i] for i in range(1 << l - 1)]
        # these will inherit the first operand's field, which is at least as large as the second; so no need to upcast.
        self._encode_helper(
            folded,
            output,
            l - 1,
            alpha_level,
            alpha_idx << 1 | 1,
            alpha_length + 1,
            alpha_level if 1 << alpha_level >= alpha_length + 2 else alpha_level + 1,
            beta_idx if alpha_length == 0 else beta_idx << 1 | 1,
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
        initial_dimension = self.log_h + self.log_inv_rate
        indeterminates_needed = ceil(log2(initial_dimension))

        unpacked = [Elem1bFP.zero()] * (1 << self.log_h + self.log_inv_rate)
        unpacked[0] = output[0]  # kill the 0 case right away, which is degenerate
        unpacked[1] = output[1]
        # for reasons that wind up being very hard to explain, it works out best with the indexing to do this---
        # i.e., to handle not just 0 but also 1 as a special case. for a baby version of this phenomenon,
        # see the constructor above (i.e., the downcasting). note that β₀ needs to get excluded there.
        # but β₀ generates both 0 and 1. the actual reasoning is tricker to explain, but leave it at that for now.

        for iota in range(indeterminates_needed):
            for i in range(1 << iota):
                subspace = 1 << iota | i
                # `subspace`: we are now going to unpack all values whose final (i.e., unpacked!) indices take the form
                # 1XX.....XX, with `subspace` many Xs. in other words, β_{subspace} + lower-order terms.
                if subspace >= initial_dimension:
                    break
                beta_bits = subspace - 1 - iota
                for j in range(1 << beta_bits):
                    j_copy = j
                    index = 1 << subspace
                    for k in range(subspace):
                        if not is_power_of_two(subspace - k):
                            index |= (j_copy & 1) << k
                            j_copy >>= 1
                    value = levels[iota + 1](
                        sum(output[1 << subspace | j << iota + 1 | k].value << k for k in range(1 << iota + 1))
                    )
                    for _ in range(1 << iota + 1):  # galois loop
                        unpacked[index] = value
                        index = self._square_in_coordinates(index)
                        value = value.square()
        return unpacked
