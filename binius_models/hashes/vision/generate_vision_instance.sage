# (C) 2024 Irreducible Inc.

# This script is based on https://github.com/KULeuven-COSIC/Marvellous/blob/master/instance_generator.sage

from hashlib import shake_256
from typing import Optional


def is_power_of_two(x: int) -> bool:
    return x & (x - 1) == 0


class BiniusTower:
    def __init__(self, degree: int):
        assert degree & (degree - 1) == 0, "degree must be a power of 2"

        self.degree = degree
        self.n = log(degree, 2)
        self.bytelen = degree // 8
        hexlen = (self.degree + 3) // 4
        self.fmt = f"{{:#0{hexlen + 2:d}x}}"

        R = PolynomialRing(GF(2), self.n, order="lex", names=[f"x{i}" for i in range(self.n - 1, -1, -1)])

        ideal = []
        last = R.one()
        for i in range(self.n - 1, -1, -1):  # vars are in reversed order
            ideal.append(R.gens()[i] ^ 2 + last * R.gens()[i] + R.one())
            last = R.gens()[i]
        self.S = R.quotient(ideal)

        # 𝔽₂-basis of S
        self.monomials = [self.S.zero()] * self.degree
        self.monomials[0] = self.S.one()
        for i in range(self.n):
            for j in range(1 << i):
                idx0 = j << self.n - i
                idx1 = idx0 | 1 << self.n - i - 1
                self.monomials[idx1] = self.monomials[idx0] * self.S.gens()[i]

    def random(self):
        return sum((monomial for monomial in self.monomials if randrange(2)), self.S.zero())

    def from_vector(self, vector):
        return sum((self.monomials[i] for i in range(len(vector)) if vector[i]), self.S.zero())

    def to_vector(self, element):
        return [1 if monomial in element.monomials() else 0 for monomial in self.monomials]

    def from_integer(self, integer):
        return sum((monomial for (i, monomial) in enumerate(self.monomials) if integer >> i & 0b1), self.S.zero())

    def to_integer(self, element):
        vector = self.to_vector(element)
        result = 0
        for i in range(self.degree):
            if vector[i]:
                result += 1 << i
        return result

    def to_hex(self, element):
        return self.fmt.format(self.to_integer(element))

    def from_bytes(self, bytes, byteorder="big"):
        assert len(bytes) == self.degree // 8, "lengths dont match"
        return self.from_integer(int.from_bytes(bytes, byteorder))

    def to_bytes(self, element, byteorder="big"):
        return self.to_integer(element).to_bytes(self.degree // 8, byteorder)

    def primitive_element(self):
        return self._get_multiplicative_generator()

    def _get_multiplicative_generator(self):
        """Returns the smallest viable multiplicative generator"""
        for i in range(2 ^ (self.degree / 2), 2 ^ self.degree):
            result = self.from_integer(i)
            if self.is_generator(result):
                return result

    def is_generator(self, element):
        if element == self.from_integer(0):
            return False
        return not any(
            (element ^ ((2 ^ self.degree - 1) // factor)).is_one()
            for (factor, multiplicity) in factor(2 ^ self.degree - 1)
        )


class Shaker:
    """Provides a pool of randomness from a seed"""

    def __init__(self, seed: bytes, buffer_size: int = 2**27) -> None:
        self.data = shake_256(seed).digest(int(buffer_size))

    def __call__(self, n: int) -> bytes:
        """Returns n bytes of randomness"""
        if len(self.data) < n:
            raise BufferError("Not enough randomness, increase pool size and run again.")

        ret, self.data = self.data[:n], self.data[n:]
        return ret


class Vision:

    def __init__(self, security_level: int, field_size: int, state_elem_num: int, mds_field_size: Optional[int] = None):
        self.security_level = security_level
        self.n = field_size
        self.m = state_elem_num
        self.mds_n = mds_field_size if mds_field_size is not None else field_size
        self.field = BiniusTower(self.n)
        self.mds_field = BiniusTower(self.mds_n)

        self.r = (field_size * state_elem_num - 2 * security_level) // field_size  # rate size
        self.c = 2 * security_level // field_size  # capacity size

        assert self.r + self.c == self.m, "rate and capacity don't fit the state"
        assert (
            field_size * state_elem_num >= security_level
        ), "state size (n) must be at least as large as security level"
        assert is_power_of_two(field_size), "field size must be a power of 2"
        assert state_elem_num % 3 == 0 and is_power_of_two(
            state_elem_num // 3
        ), "state elem num must be 3 times a power of 2"
        assert self.m < self.n, "state_elem_num must be smaller than field size"
        assert self.c * self.field.bytelen > 8, "capacity should be able to fit 64 bits"

        self.round_n = 8  # max(10, 2 * ceil((1.0 * security_level + self.m + 8) / (8 * self.m)))

        self.mds = self._generate_mds_matrix()
        # To print the matrix, uncomment the following line
        # self._print_matrix(self.mds, self.m, self.m)

        self.b, self.b_inv, self.initial_constant, self.constants_matrix, self.constants_constant, self.random_state = (
            self._sample_parameters()
        )

    def sbox(self, x, pi: int = 0):
        """Apply the sbox to a state element x.
        pi = 0: even sbox, pi_0
        pi = 1: odd sbox, pi_1
        """
        poly = self.b_inv if pi == 0 else self.b
        x_inv = x ^ (2 ^ self.field.degree - 2)
        return self.evaluate_affine_linearized_polynomial(x_inv, poly)

    def update_key_schedule(self, key: vector) -> None:
        self.key_schedule = []

        key_injection = self.initial_constant
        state = key + key_injection

        self.key_schedule.append(state)

        for r in range(2 * self.round_n):
            key_injection = self.constants_matrix * key_injection + self.constants_constant

            state = vector([self.sbox(x, r % 2) for x in state])
            state = self.mds * state + key_injection
            self.key_schedule.append(state)

    def encrypt(self, plaintext: vector) -> vector:
        """Encrypts one block of data, ECB mode encrypt."""
        state = plaintext + self.key_schedule[0]

        for r in range(2 * self.round_n):
            state = vector([self.sbox(x, r % 2) for x in state])
            state = self.mds * state + self.key_schedule[1 + r]

        return state

    def sponge(self, message: list) -> vector:
        # To actually do the sponge like construction we set the bytelength (as little endian) of the input message
        # (before padding) as the first n capacity elements where n is 8 / field.bytelength i.e the number of field
        # elements that can fit a 64 bit integer. We zero pad the message at the end until we have a multiple of rate.
        # Then for each of the rate chunks we overwrite the rate portion of the state with message and run the permutation
        # function, note that in this implementation we are not overwriting the capacity portion with the permutation
        # from the previous state.
        zero = vector([self.field.from_integer(0) for _ in range(self.m)])
        self.update_key_schedule(zero)

        state = copy(zero)
        message_bytelen = int(len(message) * self.field.bytelen).to_bytes(8, "little")
        # N.B: encode the message lenght at the begining of the capacity
        for i, left in enumerate(range(0, len(message_bytelen), self.field.bytelen)):
            state[self.r + i] = self.field.from_bytes(message_bytelen[left : left + self.field.bytelen], "little")

        if len(message) % self.r:
            message += zero[: self.r - len(message) % self.r]

        for left in range(0, len(message), self.r):
            rate = vector(message[left : left + self.r])
            state[: len(rate)] = rate
            state = self.encrypt(state)
        # returns the whole state, not only state[:self.r]
        return state

    def evaluate_affine_linearized_polynomial(self, point, polynomial):
        return polynomial[-1] + sum(point ^ (2 ^ i) * polynomial[i] for i in range(len(polynomial) - 1))

    def _print_matrix(self, A, m, n):
        print("------------------------")
        X = [[self.mds_field.to_integer(A[i, j]) for j in range(n)] for i in range(m)]
        for i in range(m):
            print(", ".join([str(X[i][j]) for j in range(n)]))
        print("------------------------")

    def _generate_mds_matrix(self):
        # Recall the Novel Polynomial Basis from [LCH14]
        # we fix a binary field K with 𝔽₂-basis β₀, … βᵣ₋₁, say.
        # for each j ∈ {0, … , 2ʳ − 1}, we define ωⱼ := j₀ ⋅ b₀ + ⋯ + jᵣ₋₁ ⋅ βᵣ₋₁, where (j₀, …, jᵣ₋₁) are j's bits.

        # writing Uᵢ :=〈β₀, … βᵢ₋₁〉for the i-dimensional 𝔽₂-subspace generated by the first i basis elements,
        # we set Wᵢ(X) := ∏_{u ∈ Uᵢ} (X − u). subspace polynomial of deg 2ⁱ; its evaluation map Wᵢ : K → K is 𝔽₂-linear
        # Ŵᵢ(X) := Wᵢ(X) / Wᵢ(βᵢ) is its normalized variant; moreover satisfies Ŵᵢ(βᵢ) = 1. and is also 𝔽₂-linear

        # finally, for each j ∈ {0, … , 2ʳ − 1} we set Xⱼ(X) = ∏ᵢ₌₀ʳ⁻¹ (Ŵᵢ(X))^(jᵢ); w/ (j₀, …, jᵣ₋₁) again j's bits
        # since each Xⱼ(X) is of degree j, the set (X₀(X), … , X_{2ʳ−1}(X)) yields a K-basis of K[X]^{< 2ʳ}
        log_m = self.m.bit_length()

        # ultimately, U[i][j] will contain Wᵢ(βⱼ), for each i ∈ {0, … ⌈ log m ⌉ } and j ∈ {0, … ⌈ log m ⌉ + 1}.
        # this information alone will be enough to compute Wᵢ(ωⱼ) for each j ∈ {0, …, 2 * log m - 1},
        # using merely some additions, since the Wᵢs are 𝔽₂-linear (in particular, additively homomorphic)
        U = [[self.mds_field.from_integer(1 << j) for j in range(log_m + 1)]]
        for i in range(1, log_m):
            # compute the row Wᵢ(β₀), … , Wᵢ(β_{⌈ log m ⌉ + 1}), given the respective values of Wᵢ₋₁ on these points.
            # we use the recursive identity Wᵢ(X) = Wᵢ₋₁(X) ⋅ (Wᵢ₋₁(X) + Wᵢ₋₁(βᵢ₋₁)) to do this efficiently.
            U.append([U[i - 1][j] * (U[i - 1][j] + U[i - 1][i - 1]) for j in range(log_m + 1)])

        for i in range(log_m):  # normalize everything (see above).
            normalization_constant = self.mds_field.from_integer(1) / U[i][i]
            U[i] = [U[i][j] * normalization_constant for j in range(log_m + 1)]

        # expand horizontally: W[i][j] will contain Ŵᵢ(ωⱼ) for each i ∈ {0, … ⌈ log m ⌉ } and j ∈ {0, …, 2 * log m - 1}
        # as explained above, we can do this only using additions, having computed the U[i][j]s.
        W = []
        for i in range(log_m):
            W_i = [self.mds_field.from_integer(0)]
            for j in range(log_m + 1):  # standard binary expansion trick: W_i will contain all subset sums of U[i].
                W_i += [W_i[k] + U[i][j] for k in range(1 << j)]
            W.append(W_i[: 2 * self.m])

        # expand vertically: X[j][i] will contain Xᵢ(ωⱼ) for each i ∈ {0, … m - 1} and j ∈ {0, …, 2 * log m - 1}.
        # we can again compute these from the Ŵᵢ(ωⱼ)s above using a binary expansion; now multiplying instead of adding
        # indeed, this is literally the definition of the Xᵢs.
        X = []
        for j in range(2 * self.m):
            X_j = [self.mds_field.from_integer(1)]
            for i in range(log_m):  # standard binary expansion, now with multiplying instead of adding
                X_j += [X_j[k] * W[i][j] for k in range(1 << i)]
            X.append(X_j[: self.m])

        # precisely because the evaluation of a polynomial (expressed w.r.t. novel basis) is a Reed–Solomon encoding,
        # multiplication by the below matrix gives us that Reed–Solomon encoding in matrix form. its rate is 1/2.
        # i.e., it's the matrix which takes the novel-basis coefficients of a polynomial of degree < m,
        # and returns its evaluations over the domain (ω₀, … , ω₂ₘ₋₁).
        # note that we use the "row convention": encoding is the mult of a row-vector on the right by a wide matrix.
        G = matrix([[X[j][i] for j in range(2 * self.m)] for i in range(self.m)])

        # by performing rref on G, we obtain a systematic version of the same code.
        # this code differs from the one above by precomposition with a K-isomorphism on the message space.
        # indeed, RREF simply amounts to left-multiplying the m × 2 ⋅ m matrix by an m × m invertible matrix.
        # the result of RREF has the identity as its left-hand half, and our desired MDS matrix on the right.
        # indeed, one definition of MDS matrix is simply the "nonsystematic" part of a systematic MDS code of rate 1/2.
        # another way of looking at it: it's the extrapolation matrix, which takes the values of some polynomial
        # of degree less than m on the set ω₀, … , ωₘ₋₁, and returns the evals of the same poly on ωₘ, … , ω₂ₘ₋₁.
        temp = G.rref()

        # Sanity-Check Assertions
        for i in range(self.m):
            for j in range(2 * self.m):
                num = temp[i, j].numerator()
                d = temp[i, j].denominator()
                assert self.mds_field.to_integer(d) == 1, "some denominator is not 1"
                if i == j:
                    assert self.mds_field.to_integer(num) == 1, "some diagonal element is not 1"
                elif j < self.m:
                    assert self.mds_field.to_integer(num) == 0, "some off-diagonal element is not 0"

        mds_matrix_transpose = matrix([[temp[i, j].numerator() for j in range(self.m, 2 * self.m)] for i in range(self.m)])
        mds_matrix = matrix([[mds_matrix_transpose[j, i] for j in range(self.m)] for i in range(self.m)])
        return mds_matrix

    def _sample_parameters(self):
        shaker = Shaker(b"Information is measured in bits.")
        zero = self.field.from_integer(0)

        # free term b_{-1} must be a generator (not in a subfield)
        free_term = zero
        while not self.field.is_generator(free_term):
            free_term = self.field.from_bytes(shaker(self.field.bytelen))

        # invertible, affine linearized polynomial b_0 * x + b_1 * x^2 + b_2 * x^4
        coefficients = [zero for _ in range(3)]

        # b_0, b_1, b_2 must define an invertible linearized polynomial
        while not self._linearized_is_invertible(coefficients):
            coefficients = [zero for _ in range(3)]
            for i in range(3):
                while not self.field.is_generator(coefficients[i]):
                    coefficients[i] = self.field.from_bytes(shaker(self.field.bytelen))

        # invertible, affine linearized polynomial B = b_0 * x + b_1 * x^2 + b_2 * x^4 + b_{-1}
        B = coefficients + [free_term]

        # B_inv is an affine inverse of B
        B_inv = self._affine_inverse(B)

        # any m random generators will do
        initial_constant = vector([zero for _ in range(self.m)])
        for i in range(self.m):
            while not self.field.is_generator(initial_constant[i]):
                initial_constant[i] = self.field.from_bytes(shaker(self.field.bytelen))

        # m^2 generators must define an invertible matrix
        mat_constant = matrix([[zero for _ in range(self.m)] for _ in range(self.m)])

        for i in range(self.m):
            for j in range(self.m):
                while not self.field.is_generator(mat_constant[i, j]):
                    mat_constant[i, j] = self.field.from_bytes(shaker(self.field.bytelen))

        while not mat_constant.is_invertible():
            for i in range(self.m):
                for j in range(self.m):
                    while not self.field.is_generator(mat_constant[i, j]):
                        mat_constant[i, j] = self.field.from_bytes(shaker(self.field.bytelen))

        # any m random generators will do
        constants_constant = vector([zero for _ in range(self.m)])
        for i in range(self.m):
            while not self.field.is_generator(constants_constant[i]):
                constants_constant[i] = self.field.from_bytes(shaker(self.field.bytelen))

        # random rate for testing
        random_state = vector([self.field.from_bytes(shaker(self.field.bytelen)) for _ in range(self.m)])

        return B, B_inv, initial_constant, mat_constant, constants_constant, random_state

    def _linearized_is_invertible(self, polynomial):
        deg = self.field.degree
        mat = matrix.circulant(polynomial[:3] + [self.field.from_integer(0)] * (deg - 3))

        for i in range(1, deg):
            for j in range(deg):
                mat[i, j] = mat[i, j] ^ (2 ^ i)

        return mat.is_invertible()

    def _affine_inverse(self, polynomial):
        deg = self.field.degree
        d = len(polynomial) - 1
        mat = matrix.circulant(polynomial[:d] + [self.field.from_integer(0)] * (deg - d))

        for i in range(1, deg):
            for j in range(0, deg):
                mat[i, j] = mat[i, j] ^ (2 ^ i)

        mat = mat.inverse_of_unit()

        coefficients = [mat[0, i] for i in range(deg)]
        free_term = sum(coefficients[i] * polynomial[d] ^ (2 ^ i) for i in range(deg))

        return coefficients + [free_term]


mark32b = Vision(128, 32, 24, 8)
