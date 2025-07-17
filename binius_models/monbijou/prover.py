import galois

from .constraint_system import ConstraintSystem
from .field import field


class Witness:
    def __init__(self, data: list[int]):
        self.data = data


class Prover:
    def _precompute_barycentric_weights(self):
        self.w = []
        for i in range(64):
            product = field(1)
            for j in range(64):
                if j == i:
                    continue
                product *= field(i ^ j)
            self.w.append(product**-1)

    def __init__(self, constraint_system: ConstraintSystem, witness: Witness):
        assert len(witness.data) == 1 << constraint_system.l_words
        assert all(witness.data[i] in range(1 << 64) for i in range(1 << constraint_system.l_words))
        self.constraint_system = constraint_system

        # begin populating inner
        self.a = [0] * (1 << constraint_system.l_and)
        self.b = [0] * (1 << constraint_system.l_and)
        self.c = [0] * (1 << constraint_system.l_and)

        for i, (I_a, I_b, I_c) in enumerate(constraint_system.and_constraints):
            for j, op, s in I_a:
                self.a[i] ^= op(witness.data[j], s)
            for j, op, s in I_b:
                self.b[i] ^= op(witness.data[j], s)
            for j, op, s in I_c:
                self.c[i] ^= op(witness.data[j], s)

        assert all(self.a[i] & self.b[i] == self.c[i] for i in range(1 << constraint_system.l_and))

    def _l_tilde(self, r_zhat_prime):
        # computes the list L̃(r'_ẑ, i), for i ∈ {0, … , 63}. the standard barycentric approach is to realize that,
        # for each i, L̃(r'_ẑ, i) == ∏ⱼ(r'_ẑ – j) ⋅ [ w[i] / (r'_ẑ – i) ], where w[i] are the barycentric weights.
        # the left-hand product ∏ⱼ( r'_ẑ – j) doesn't depend on i, so we can compute it once, using n – 1 mults,
        # and then finish the job using n mults and n divisions. total n - 1 + 2 ⋅ n == 3 ⋅ n - 1 mults and n inversions

        # we are going to be slightly cleverer. for each i, we want the all-but-one product p[i] ≔ ∏_{j ≠ i} (r'_ẑ – j)
        # if we have those, we can finish the job w/ n further mults, by setting L̃(r'_ẑ, i) ≔ p[i] ⋅ w[i] for each i.
        # we can get the all-but-one products with 2 ⋅ n space overhead by precomputing the prefix and suffix products
        # when all is said and done, we get 2 ⋅ (n – 1) + 2 ⋅ n = 4 ⋅ n - 2 mults, so we replaced invs --> mults
        # the problem size is small---but see later whether we end up needing this in any "hot" parts of the code?
        prefixes = [field(1)] * 64
        suffixes = [field(1)] * 64
        for i in range(1, 64):
            prefixes[i] = prefixes[i - 1] * (r_zhat_prime - field(i - 1))
            suffixes[64 - 1 - i] = suffixes[64 - i] * (r_zhat_prime - field(64 - i))
        result = [field(0)] * 64
        for i in range(64):
            result[i] = prefixes[i] * suffixes[i] * self.w[i]
        return result

    def _directly_evaluate(self, r_zhat_prime: galois.FieldElement, r_x_prime: list[galois.FieldElement]):
        # directly computes and coughs up the values of â(Ẑ, X), b̂(Ẑ, X) and ĉ(Ẑ, X) on given inputs r'_ẑ, r'_x.
        # verifier can use the resulting things as the claims α_a, α_b, α_c to feed into the shift reduction
        # in practice, these values will probably "pop out" for free at the end of the prover's univariate skip.
        # as for the verifier, my guess is that the prover will nondeterministically send them to the verifier,
        # so that the verifier can wrap up the univariate skip. pending the correctness of the supplied values.
        # the verifier will thus be done with univariate skip, and can turn his attention to the shift reduction.

        # this method can be viewed as a stand-in for the prover's univariate skip, in terms of the state we're left in.
        # will take this opportunity to memoize l_tilde ≔ L̃(r'_ẑ, i); the prover would compute this at the end of skip
        assert len(r_x_prime) == self.constraint_system.l_and
        self.l_tilde = self._l_tilde(r_zhat_prime)

        def extrapolate_bitstring(value):
            # interpreting `value` as the (bit-valued!) evaluations of some univariate polynomial over the domain D,
            # returns that polynomial's value at r'_x. because we precomputed L̃(r'_ẑ, i), just now need a subset sum.
            return sum((self.l_tilde[i] for i in range(64) if value >> i & 1), field(0))

        def evaluate(m):
            univariate_partial = [extrapolate_bitstring(m[i]) for i in range(1 << self.constraint_system.l_and)]
            for i in range(self.constraint_system.l_and):
                for j in range(1 << self.constraint_system.l_and - i - 1):
                    univariate_partial[j] = univariate_partial[j << 1] + r_x_prime[i] * (
                        univariate_partial[j << 1 | 1] - univariate_partial[j << 1]
                    )
            return univariate_partial[0]

        return (evaluate(self.a), evaluate(self.b), evaluate(self.c))

    def initiate_first_phase(
        self, lambda_a: galois.FieldElement, lambda_b: galois.FieldElement, lambda_c: galois.FieldElement
    ):
        h_sll = []
        h_srl = []
        h_sra = []

        # prepare h_op(j, s) for each j, s
        for j in range(64):
            h_sll.append(self.l_tilde[j:] + [field(0)] * j)
            h_srl.append(list(reversed(self.l_tilde[: j + 1])) + [field(0)] * (64 - 1 - j))
            h_sra.append(list(reversed(self.l_tilde[: j + 1])) + [field(0)] * (64 - 1 - j))
        for s in range(1, 64):
            h_sra[63][s] += h_sra[63][s - 1]

        for j in range(64):
            for s in range(64):
                g_op_i_j = field(0)
                for m, lambda_m in [(self.a, lambda_a), (self.b, lambda_b), (self.c, lambda_c)]:
                    g_op_i_j += lambda_m * field(0)
                h_sll[j][s] *= g_op_i_j
