from math import ceil, log2
from typing import Generic, TypeVar

import numpy as np

from ..finite_fields.tower import BinaryTowerFieldElem
from ..utils.utils import bits_mask, is_bit_set, is_power_of_two

F = TypeVar("F", bound=BinaryTowerFieldElem)


class AdditiveNTT(Generic[F]):
    # this is essentially a binary-field Reed–Solomon encoder.
    # internally its key subroutine is a binary NTT following the "novel polynomial basis paper" [LCH14]
    # see also Frobenius Additive NTT, [LCK+18], sections 1 – 2, for helpful background info
    # given a vector of coefficients in the novel polynomial basis, (zero-pads and) evaluates polynomial on a subspace

    # field of coefficients 𝔽_{2ʳ} == 𝔽_{2^{2ⁿ}}. note that we only allow coefficient fields of power-of-2 deg,
    # unlike in LCH14, which allows arbitrary r. the reason is that we will rep. them as a FAST tower internally.
    def __init__(self, field: type[F], max_log_h: int, log_inv_rate: int, skip_rounds: int = 0) -> None:
        if field.field.degree < max_log_h + log_inv_rate + skip_rounds:
            raise ValueError("field degree must be at least log_h + log_inv_rate")

        self.field = field
        self.max_log_h = max_log_h
        self.log_inv_rate = log_inv_rate
        self._precompute_constants(skip_rounds)

    def _precompute_constants(self, skip_rounds: int = 0) -> None:
        self.constants: list[list[F]] = [[]]
        # self.constants is a "triangular" array containing various precomputed constants for use later.
        # it's not quite triangular, since it's rectangular (wider than it is tall); exact indexing is as follows:
        # for each 0 ≤ i < log_h, 0 ≤ j < log_h + log_inv_rate - 1 - i (so log_h + log_inv_rate - 1 - i elems in row i),
        # self.constants[i][j] = sᵢ(vᵢ₊₁₊ⱼ), i.e., the image of the iᵗʰ subspace function on the Cantor vector vᵢ₊₁₊ⱼ.
        # so in practice, this is a list of log_h lists, where the iᵗʰ list has log_h + log_inv_rate - 1 - i elements.
        for i in range(self.max_log_h + skip_rounds + self.log_inv_rate):
            self.constants[0].append(self.field(1 << i))
        for i in range(1, self.max_log_h + skip_rounds):
            self.constants.append([])
            for j in range(self.max_log_h + skip_rounds + self.log_inv_rate - i):
                self.constants[i].append(self._s(self.constants[i - 1][j + 1], self.constants[i - 1][0]))
        for i in range(self.max_log_h + skip_rounds):
            for j in range(self.max_log_h + skip_rounds + self.log_inv_rate - i - 1, -1, -1):
                self.constants[i][j] /= self.constants[i][0]
        self.constants = self.constants[skip_rounds:]

    def _s(self, element: F, constant: F) -> F:
        return element.square() + constant * element

    def _naive_encode(self, input: list[F]) -> list[F]:  # ONLY FOR TESTING, slow alg
        # we're going to naïvely convert input from novel to normal polynomial basis;
        # then evaluate the input over the entire extended domain naïvely. it's going to be slow.
        assert len(input) <= 1 << self.max_log_h, "input is too large."
        assert is_power_of_two(len(input)), "input length must be a power of 2"
        log_h = len(input).bit_length() - 1
        s_polys = [[self.field.one()]]  # sparse! include log-many coordinates
        normalization_constants = [self.field.one()]
        for i in range(1, log_h):
            s_polys.append([self.field.zero()] * (i + 1))
            for j in range(i, 0, -1):
                s_polys[i][j] = s_polys[i - 1][j - 1].square()
            for j in range(i - 1, -1, -1):
                s_polys[i][j] += normalization_constants[i - 1] * s_polys[i - 1][j]
            constant = self.field.zero()  # will equal W_i(1 << i). do this naïvely, to test...
            accum = self.constants[0][i]
            for j in range(i + 1):
                constant += s_polys[i][j] * accum
                accum = accum.square()
            normalization_constants.append(constant)
        for i in range(1, log_h):  # normalize all w_polys
            for j in range(i + 1):
                s_polys[i][j] /= normalization_constants[i]

        input_monomial = [self.field.zero()] * len(input)  # coefficients of input, w.r.t monomial basis.
        X_poly = [self.field.one()] + [self.field.zero()] * (len(input) - 1)

        def _X_assembler(X_poly, level, index):  # recursive helper function, exploit some closures
            if level == log_h:
                for i, coefficient in enumerate(X_poly[: index + 1]):
                    input_monomial[i] += input[index] * coefficient
                return
            _X_assembler(X_poly, level + 1, index)
            X_poly_times_s = [self.field.zero()] * len(input)  # begin computation of X_poly * s_polys[level]
            # s_polys[level] is sparse, of degree 2^{level}. only the coefficients indexed {0, ..., level} are nonzero.
            # X_poly is of degree at most `index`; i.e., only its coefficients {0, ..., index} can be nonzero.
            for i, coefficient in enumerate(s_polys[level][: level + 1]):  # this thing is sparse!
                for j in range(index + 1):
                    X_poly_times_s[j + (1 << i)] += coefficient * X_poly[j]
            _X_assembler(X_poly_times_s, level + 1, index | 1 << level)

        _X_assembler(X_poly, 0, 0)
        # claim: input_monomial now contains the representation of input in the monomial basis. time to evaluate...
        result = [self.field.zero()] * (len(input) << self.log_inv_rate)
        for i in range(1 << log_h + self.log_inv_rate):  # for each evaluation point x in the target domain...
            power_of_x = self.field.one()  # viewed as a tower element...
            value = sum(
                (self.constants[0][j] for j in range(log_h + self.log_inv_rate) if is_bit_set(i, j)), self.field.zero()
            )
            for j in range(1 << log_h):  # for each power j...
                result[i] += input_monomial[j] * power_of_x
                power_of_x *= value
        return result

    def _calculate_twiddle(self, i: int, j: int, coset: int, log_h: int) -> F:
        assert j in range(1 << log_h - 1 - i)
        return sum(
            (
                self.constants[i][k + 1]
                for k in range(self.max_log_h + self.log_inv_rate - 1 - i)
                if is_bit_set(coset << log_h - 1 - i | j, k)
            ),
            self.field.zero(),
        )

    def _transform(self, input: list[F], coset: int) -> list[F]:
        assert len(input) <= 1 << self.max_log_h, "input is too large."
        assert is_power_of_two(len(input)), "input length must be a power of 2"
        log_h = len(input).bit_length() - 1
        assert coset in range(1 << self.log_inv_rate + self.max_log_h - log_h), "coset must be in range"
        # see Alg. 2 of Frobenius Additive FFT, [LCK+18]
        # coset will be an int ∈ {0, ..., (1 << log_inv_rate) - 1}, controlling which coset shift we're in
        result = input.copy()

        for i in range(log_h - 1, -1, -1):  # stage of butterfly, moving from left to right
            for j in range(1 << log_h - 1 - i):  # "block" of butterfly within ith stage; moving top to down
                # the following line of code gives us a twiddle factor, suitable for use in this block.
                # essentially, we need to perform a certain subset sum of our precomputed constants, where the
                # subsets at hand are determined by the bits of our coset and the bits of our butterfly block j.
                twiddle = self._calculate_twiddle(i, j, coset, log_h)
                # at this point, `twiddle` is the correct twiddle; we will use it for each butterfly wire below
                for k in range(1 << i):  # indexes the actual lines we're taking; all same constant
                    idx0 = j << i + 1 | k
                    idx1 = idx0 | 1 << i
                    result[idx0] += twiddle * result[idx1]
                    result[idx1] += result[idx0]
        return result

    def _inverse_transform(self, input: list[F], coset: int) -> list[F]:
        assert len(input) <= 1 << self.max_log_h, "input is too large."
        assert is_power_of_two(len(input)), "input length must be a power of 2"
        log_h = len(input).bit_length() - 1
        assert coset in range(1 << self.log_inv_rate + self.max_log_h - log_h), "coset must be in range"
        result = input.copy()

        # Notice that we start with the largest butterfly blocks and move to the smallest
        for i in range(log_h):
            for j in range(1 << log_h - 1 - i):
                twiddle = self._calculate_twiddle(i, j, coset, log_h)
                for k in range(1 << i):
                    idx0 = j << i + 1 | k
                    idx1 = idx0 | 1 << i
                    # Notice these two lines are swapped compared to the forward transform
                    result[idx1] += result[idx0]
                    result[idx0] += twiddle * result[idx1]
        return result

    def encode(self, input: list[F]) -> list[F]:
        assert len(input) <= 1 << self.max_log_h, "input is too large."
        assert is_power_of_two(len(input)), "input length must be a power of 2"
        return sum((self._transform(input, coset) for coset in range(1 << self.log_inv_rate)), [])


class FourStepAdditiveNTT(Generic[F]):
    def __init__(self, field: type[F], log_h: int, log_inv_rate: int) -> None:
        if field.field.degree < log_h + log_inv_rate:
            raise ValueError("field degree must be at least log_h + log_inv_rate")

        self.field = field
        self.log_h = log_h
        self.log_inv_rate = log_inv_rate

        self.inner_ntt_log_len = self.log_h // 2
        self.outer_ntt_log_len = self.log_h - self.inner_ntt_log_len
        self.inner_ntt_len = 1 << self.inner_ntt_log_len
        self.outer_ntt_len = 1 << self.outer_ntt_log_len

        # rig up "outer" and "inner" NTTs. each thinks it's just an NTT internally, but will be used in our two stages.
        # the two NTTs' matrices of constants are the lower and upper halves, respectively, of the standard NTT matrix
        # for the lower half, we have the same log_inv_rate, and half as many columns for h; i.e., log_h >>= 1.
        # for the upper half, we again have log_h >>= 1, but also, log_inv_rate += log_h >> 1; i.e., the rate is higher
        # in any case everything works out as it should; this is described in detail in the draw.io board
        self.outer_ntt = AdditiveNTT(field, self.outer_ntt_log_len, log_inv_rate, skip_rounds=self.inner_ntt_log_len)
        self.inner_ntt = AdditiveNTT(field, self.inner_ntt_log_len, self.outer_ntt_log_len + log_inv_rate)

    def _transpose(self, input: list[list[F]]) -> list[list[F]]:
        return [[input[j][i] for j in range(self.inner_ntt_len)] for i in range(self.outer_ntt_len)]

    def _transform(self, input: list[F], coset: int) -> list[F]:
        # `coset` ranges from {0, ..., (1 << log_inv_rate) - 1}.
        # transform input into a matrix, in column-major order
        input_mat = [
            [input[i + j * self.inner_ntt_len] for j in range(self.outer_ntt_len)] for i in range(self.inner_ntt_len)
        ]
        for i in range(self.inner_ntt_len):
            input_mat[i] = self.outer_ntt._transform(input_mat[i], coset)
        input_mat = self._transpose(input_mat)
        for i in range(self.outer_ntt_len):
            input_mat[i] = self.inner_ntt._transform(input_mat[i], coset << self.outer_ntt_log_len | i)
        return [
            input_mat[i >> self.inner_ntt_log_len][i & bits_mask(self.inner_ntt_log_len)]
            for i in range(1 << self.log_h)
        ]

    def encode(self, input: list[F]) -> list[F]:
        assert len(input) == 1 << self.log_h
        return sum((self._transform(input, coset) for coset in range(1 << self.log_inv_rate)), [])


class CantorAdditiveNTT(AdditiveNTT[F]):
    # subclass corresponding to the case that we're in the Cantor basis, each Ŵᵢ(ω_{2ⁱ)} == 1,
    # and we can prune a few steps away from _precompute_constants.
    def _precompute_constants(self, skip_rounds: int = 0) -> None:
        self.constants: list[list[F]] = [[]]
        for i in range(self.max_log_h + skip_rounds + self.log_inv_rate):
            self.constants[0].append(self.field(1 << i))
        for i in range(1, self.max_log_h + skip_rounds):
            self.constants.append([])
            for j in range(self.max_log_h + skip_rounds + self.log_inv_rate - i):
                self.constants[i].append(self._s(self.constants[i - 1][j + 1], self.field.one()))
        self.constants = self.constants[skip_rounds:]


class FancyAdditiveNTT(AdditiveNTT[F]):
    # for our S⁽⁰⁾, we're going to take the image in the Fan–Paar field OF the set < 1, 2, 4, ... > in the FAST field.
    # and we can prune a few steps away from _precompute_constants.
    def _field_to_column(self, element: F, iota: int) -> np.array:
        return np.array([(element.value >> i) & 1 for i in range(1 << iota)]).reshape(-1, 1)

    def _column_to_field(self, column: np.array, iota: int) -> F:
        return self.field(sum(column.tolist()[i][0] << i for i in range(1 << iota)))

    def _solve_underdetermined_system(self, products: np.array, affine_constant: np.array, iota: int) -> np.array:
        # the matrices we call this on are guaranteed to be 0 in the leftmost column, and elsewhere of full rank.
        # augmented = galois.FieldArray.row_reduce(np.hstack((products, affine_constant)))
        # return np.insert(augmented[:, -1][:-1], 0, 0).reshape(-1, 1)
        # using just the above two lines ^^^, we could one-shot this thing.
        # i am purposefully going to avoid using the RREF solver, so that this can be ported to Rust more natively.
        augmented = np.hstack((products, affine_constant))
        # RREF solver, ASSUMING that `products` is 2^ι × 2^ι, 0 in the leftmost column and elsewhere of full rank.
        # every positive column i > 0 will have a pivot in the i – 1th row, and the bottom row will be empty.
        for pivot in range(1, 1 << iota):
            # the pivot is going to wind up being in the `pivot - 1`th row
            new_pivot_row = list(augmented[:, pivot][pivot - 1 :]).index(1) + pivot - 1
            augmented[[pivot - 1, new_pivot_row]] = augmented[[new_pivot_row, pivot - 1]]

            for row in range(1 << iota):
                if row is not pivot - 1 and augmented[row, pivot] == 1:
                    augmented[row] ^= augmented[pivot - 1]
        return np.insert(augmented[:, -1][:-1], 0, 0).reshape(-1, 1)

    def _precompute_constants(self, skip_rounds: int = 0) -> None:
        initial_dimension = self.max_log_h + skip_rounds + self.log_inv_rate
        self.constants: list[list[F]] = [[]]
        self.constants[0].append(self.field(1))
        iota = 0
        # for each ι, this will be a 2^ι × 2^ι bit-matrix.
        # its columns will be the bit-decompositions in the FP basis of α² + α, for α varying through an 𝔽₂-basis of 𝒯_ι
        products = np.zeros((1, 1), dtype=np.uint8)
        for _ in range(ceil(log2(initial_dimension))):
            iota += 1
            # begin construction of tower level ι.
            products = np.pad(products, ((0, 1 << iota - 1), (0, 0)), mode="constant", constant_values=0)
            for j in range(1 << iota - 1):
                new_fp_vector = self.field(1 << (1 << iota - 1 | j))
                image = new_fp_vector.square() + new_fp_vector
                # Get the integer value from the field element and convert to binary column
                products = np.hstack((products, self._field_to_column(image, iota)))

            # the below is the decomposition, with respect to our basis, of the image, in our field,
            # of the FAST constant monomial X_0 ⋅ ⋯ X_{ι − 2}. put that in your pipe and smoke it!
            affine_constant = self._field_to_column(self.constants[0][(1 << iota - 1) - 1], iota)

            solution = self._solve_underdetermined_system(products, affine_constant, iota)
            fast_indeterminate = self._column_to_field(solution, iota)  # == image of FAST X_{ι – 1} in the FP tower
            for j in range(min(1 << iota - 1, initial_dimension - (1 << iota - 1))):
                self.constants[0].append(self.constants[0][j] * fast_indeterminate)

            if len(self.constants[0]) == initial_dimension:
                break
        # we've gotten the top row; from this point forward, we do the usual thing
        for i in range(1, self.max_log_h + skip_rounds):
            self.constants.append([])
            for j in range(self.max_log_h + skip_rounds + self.log_inv_rate - i):
                self.constants[i].append(self._s(self.constants[i - 1][j + 1], self.field.one()))
        self.constants = self.constants[skip_rounds:]


class GaoMateerBasis(AdditiveNTT[F]):
    def _precompute_constants(self, skip_rounds: int = 0) -> None:
        # assert: self.field.degree >= self.max_log_h + self.log_inv_rate + skip_rounds
        # really we shouldn't be taking in the field; we should be defining it ourselves minimally to satisfy the above
        initial_dimension = self.max_log_h + skip_rounds + self.log_inv_rate
        self.constants: list[list[F]] = [[]]
        indeterminates_needed = ceil(log2(initial_dimension))
        self.constants[0] = [self.field.zero()] * (1 << indeterminates_needed)
        self.constants[0][(1 << indeterminates_needed) - 1] = self.field(1 << (1 << indeterminates_needed - 1))
        for i in range((1 << indeterminates_needed) - 1, 0, -1):
            self.constants[0][i - 1] = self.constants[0][i].square() + self.constants[0][i]
        for i in range(1, self.max_log_h + skip_rounds):
            self.constants.append(self.constants[0][: initial_dimension - i])  # trivial, no computation needed
        self.constants = self.constants[skip_rounds:]
