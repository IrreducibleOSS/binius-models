import random  # just to mock the verifier's queries; not cryptographically secure

from ..ips.sumcheck import Sumcheck
from ..ips.utils import Elem128b, Polynomial128
from ..ntt.additive_ntt import AdditiveNTT


class VectorOracle:
    def __init__(self) -> None:
        self.vectors: list[list[Elem128b]] = []

    def commit(self, list: list[Elem128b]):
        self.vectors.append(list)

    def query(self, index: int, position: int) -> Elem128b:
        return self.vectors[index][position]


class FRIBinius:  # (Generic[F])
    def __init__(self, field: type[Elem128b], var: int, log_rate: int):
        self.var = var
        self.log_rate = log_rate
        self.field = field
        self.additive_ntt = AdditiveNTT[Elem128b](field, self.var, log_rate)
        self.challenges: list[Elem128b] = []  # memoize the round challenges we receive from the verifier...
        self.oracle = VectorOracle()
        # ...the ONLY place we use this will be in the self.verifier_query method, which is added for testing purposes.

    def _fold(self, position: Elem128b, values: tuple[Elem128b, Elem128b], r: Elem128b) -> Elem128b:
        # see FRI-Binius, Def. 3.13
        # note: `position` and `position + 1` live in small (synthetic) subfield.
        # r lives in constant subfield.
        # implement multiplication accordingly.
        mult = [values[0], values[1]]
        mult[1] += mult[0]
        mult[0] += mult[1] * position
        return (self.field.one() + r) * mult[0] + r * mult[1]

    def _get_preimage(self, i: int, position: int):
        # writing i = self.sumcheck round and interpreting `position` as an element of {0, 1}^{ℓ + ℛ − i - 1} ≅ S⁽ⁱ⁺¹⁾,
        # returns _the 0th_ among the two elements of S⁽ⁱ⁾ sitting above `position` (the other element differs by 1).
        # we interpret position as a vector of coords w.r.t. the "canonical" 𝔽₂-basis of S⁽ⁱ⁺¹⁾, given in the paper.
        # since we chose our bases of the S⁽ⁱ⁾s well (see paper), the coordinates of the elements of the fiber
        # are precisely the same as those of `position`, with either a 0 or a 1 prepended.
        # moreover the _basis_ with respect to which said coordinates should be used as a combination is precisely
        # what we already have stored in the additive NTT (namely, the elements S⁽ⁱ⁾(βᵢ₊₁), ...,  S⁽ⁱ⁾(β_{ℓ + ℛ − 1}))!
        # note interestingly that in both that case and this, the 0th basis element, namely S⁽ⁱ⁾(βᵢ) = 1, is missing.
        # we don't need it there; nor do we need it here: we're only going to return the 0th fiber element.
        # so don't bother, simply take the positive-indexed basis vectors and use `position` as the combination vector.
        return self.additive_ntt._calculate_twiddle(i, position, 0, self.var + self.log_rate)

    def commit(self, multilinear: list[Elem128b]) -> None:
        assert len(multilinear) == 1 << self.var  # length is a power of 2
        self.multilinear = multilinear
        output = self.additive_ntt.encode(self.multilinear)
        self.oracle.commit(output)  # commit 0th

    def initialize_proof(self, r: list[Elem128b]) -> None:
        assert len(r) == self.var
        composition = Polynomial128(2, {tuple([1, 1]): Elem128b.one()})
        eq_r = [Elem128b.one()] + [Elem128b.zero()] * (len(self.multilinear) - 1)
        for i in range(self.var):
            for h in range(1 << i):
                eq_r[1 << i | h] = eq_r[h] * r[i]
                eq_r[h] -= eq_r[1 << i | h]
        # todo: is there a faster way to do sumcheck that exploits the structure of and_r?
        self.sumcheck = Sumcheck([self.multilinear, eq_r], composition)

    def advance_state(self) -> list[Elem128b]:
        return self.sumcheck.compute_round_polynomial()

    def receive_challenge(self, r: Elem128b) -> None:
        self.challenges.append(r)
        i = self.sumcheck.round
        next_round_oracle = [Elem128b.zero()] * (1 << self.var + self.log_rate - i - 1)
        for u in range(1 << self.var + self.log_rate - i - 1):  # hasn't +='d round yet
            values = (self.oracle.vectors[i][u << 1], self.oracle.vectors[i][u << 1 | 1])
            next_round_oracle[u] = self._fold(self._get_preimage(i, u), values, r)
        self.sumcheck.receive_challenge(r)
        self.oracle.commit(next_round_oracle)

    def finalize(self) -> Elem128b:
        return self.oracle.vectors[self.var][0]

    def verifier_query(self, result: Elem128b) -> None:  # either assertion failure or return none
        # performs _one round_ of the verifier's query procedure. will be repeated γ, etc., times.
        # "result" is nothing other than the prover's final FRI response that he sent in the clear to the verifier.
        # note that of course in practice this whole thing will be done by the verifier,
        # and in particular the randomness will be chosen by the verifier.
        assert self.sumcheck.round == self.var  # sumcheck has already completed
        v = random.randrange(1 << self.var + self.log_rate)  # note: this should be cryptographically random in practice
        c = self.field.zero()
        for i in range(self.var):
            values = (self.oracle.vectors[i][~(~v | 0x01)], self.oracle.vectors[i][v | 0x01])  # query at coset of v
            if i > 0:
                assert c == self.oracle.vectors[i][v]
            v >>= 1
            c = self._fold(self._get_preimage(i, v), values, self.challenges[i])
        assert c == result
