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


class SumcheckClaim:  # each ring-switching instance _outputs_ / reduces to one of these. two multilinears
    def __init__(self, multilinear: list[Elem128b], folded_eq_indicator: list[Elem128b], index: int) -> None:
        # assert len(multilinears[0]) == len(multilinears[1])  # will do so internally
        composition = Polynomial128(2, {tuple([1, 1]): Elem128b.one()})  # simple product of two multilinears
        self.sumcheck = Sumcheck([multilinear, folded_eq_indicator], composition, False)
        self.value = sum((a * eq for (a, eq) in zip(multilinear, folded_eq_indicator)), Elem128b.zero())
        # presumably in real life we will have this value sitting around somewhere, as opposed to having to compute it.
        self.next = index + 1


class SumcheckManager:  # wraps a bunch of individual sumcheck claims.
    def __init__(self, claims: list[SumcheckClaim]):
        self.round = 0
        self.claims = claims

    def initialize(self, batching_randomness: Elem128b) -> None:
        self.batching_randomness = batching_randomness

    def advance_state(self):
        round_polynomial = [Elem128b.zero()] * 2  # composition degree == 2; we are going to omit 0 as usual
        randomness = Elem128b.one()
        for claim in sorted(self.claims, key=lambda claim: claim.sumcheck.v):
            if self.round < claim.sumcheck.v:
                individual_round_polynomial = claim.sumcheck.compute_round_polynomial()
                for i in range(len(round_polynomial)):  # accumulate; otherwise add nothing.
                    round_polynomial[i] += randomness * individual_round_polynomial[i]
            randomness *= self.batching_randomness
        return round_polynomial

    def receive_challenge(self, r: Elem128b):
        for claim in self.claims:
            if self.round < claim.sumcheck.v:
                claim.sumcheck.receive_challenge(r)
        self.round += 1


class BatchedFRIBinius:  # (Generic[F])
    def __init__(self, field: type[Elem128b], log_rate: int, claims: list[SumcheckClaim]):
        self.log_rate = log_rate
        self.field = field

        self.claims = claims  # we assume it's sorted in descending order
        self.manager = SumcheckManager(claims)
        self.concatenated = []
        for claim in claims:
            self.concatenated += claim.sumcheck.multilinears[0]
        self.var = (len(self.concatenated) - 1).bit_length()
        self.concatenated += [Elem128b.zero()] * ((1 << self.var) - len(self.concatenated))  # zero-pad the rest

        self.additive_ntt = AdditiveNTT[Elem128b](field, self.var, log_rate)
        self.challenges: list[Elem128b] = []  # memoize the round challenges we receive from the verifier...
        self.oracle = VectorOracle()
        # ...the ONLY place we use this will be in the self.verifier_query method, which is added for testing purposes.

        # we are going to prepare the way for the verifier with a bit of bookkeeping trickiness.
        # this gives us a dictionary where: key == a possible round index, so in {0, ..., total_composition_vars - 1},
        # and value tells us, which index idx, into our main list of claims, is minimal, such that claims[idx].v ≤ key?
        # this turns out to be necessary below during our "piecewise reconstruction" routine. we do an on-the-fly idea
        # where this information tells us where to "start interpolating stray pieces of our piecewise thing".
        self.positions = {}
        position = len(claims)
        for i in range(self.var + 1):
            while position > 0 and claims[position - 1].sumcheck.v <= i:
                position -= 1
            self.positions[i] = position

    def _fold(self, position: Elem128b, values: tuple[Elem128b, Elem128b], r: Elem128b) -> Elem128b:
        mult = [values[0], values[1]]
        mult[1] += mult[0]
        mult[0] += mult[1] * position
        return (self.field.one() + r) * mult[0] + r * mult[1]

    def _get_preimage(self, i: int, position: int):
        return self.additive_ntt._calculate_twiddle(i, position, 0, self.var + self.log_rate)

    def commit(self) -> None:
        output = self.additive_ntt.encode(self.concatenated)
        self.oracle.commit(output)  # commit 0th

    def initialize(self, batching_randomness: Elem128b) -> None:
        self.batching_randomness = batching_randomness
        self.manager.initialize(batching_randomness)

    def advance_state(self) -> list[Elem128b]:
        return self.manager.advance_state()

    def receive_challenge(self, r: Elem128b) -> list[Elem128b]:
        self.challenges.append(r)
        i = self.manager.round
        next_round_oracle = [Elem128b.zero()] * (1 << self.var + self.log_rate - i - 1)
        for u in range(1 << self.var + self.log_rate - i - 1):  # hasn't +='d round yet
            values = (self.oracle.vectors[i][u << 1], self.oracle.vectors[i][u << 1 | 1])
            next_round_oracle[u] = self._fold(self._get_preimage(i, u), values, r)
        self.manager.receive_challenge(r)
        self.oracle.commit(next_round_oracle)
        return [self.claims[j].sumcheck.multilinears[0][0] for j in range(self.positions[i + 1], self.positions[i])]

    def finalize(self) -> Elem128b:
        return self.oracle.vectors[self.var][0]

    def verifier_query(self, result: Elem128b) -> None:  # either assertion failure or return none
        # performs _one round_ of the verifier's query procedure. will be repeated γ, etc., times.
        # "result" is nothing other than the prover's final FRI response that he sent in the clear to the verifier.
        # note that of course in practice this whole thing will be done by the verifier,
        # and in particular the randomness will be chosen by the verifier.
        v = random.randrange(1 << self.var + self.log_rate)  # note: this should be cryptographically random in practice
        c = self.field.zero()
        for i in range(self.var):
            values = (self.oracle.vectors[i][~(~v | 0x01)], self.oracle.vectors[i][v | 0x01])  # query at coset of v
            if i > 0:
                assert c == self.oracle.vectors[i][v]
            v >>= 1
            c = self._fold(self._get_preimage(i, v), values, self.challenges[i])
        assert c == result
