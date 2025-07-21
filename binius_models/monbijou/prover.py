from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem
from ..ips.sumcheck import Sumcheck
from ..ips.utils import Polynomial
from .constraint_system import ConstraintSystem, ShiftOp
from .utils import tensor_expand, univariate_expand

F = TypeVar("F", bound=FiniteFieldElem)


class Prover(Generic[F]):
    def __init__(self, field: type[F], constraint_system: ConstraintSystem, witness: list[int]):
        self.field = field
        self.constraint_system = constraint_system
        self.witness = witness
        assert len(witness) == 1 << constraint_system.l_words
        assert all(witness[i] in range(1 << 64) for i in range(1 << constraint_system.l_words))
        assert 1 << constraint_system.l_and == len(constraint_system.and_constraints)

        # begin populating inner
        self.a = [0] * (1 << constraint_system.l_and)
        self.b = [0] * (1 << constraint_system.l_and)
        self.c = [0] * (1 << constraint_system.l_and)

        # in one pass over the constraint system, both accumulate our a, b, c values and set up the "column index"
        # the a, b, c are just as in the paper, and are computed in the same way as there (by shifting and accumulating)
        # the column index is an auxiliary data structure which will be used to speed up the sumcheck's first phase.
        # the point is to store, for each value y ∈ {0, … , n_{words} – 1} indexing into the witness,
        # the set of (op, m, s)s for which the yᵗʰ column of the matrix M_{op, m}(X, y, s) is not identically zero.
        # these will be exactly the (op, m, s) for which the prover needs to update the "gacc" accumulator; see below.

        # btw, our use of sets is necessary. to see this, take the case that M_{op, m}(X, y, s) has 2 nonzero entries.
        # in this case, M[op][m][s][y] below will be the sum of exactly two entries of the tensor of r'ₓ.
        # i.e., the fact of two entries being present in that column will be accounted for during the computation of M.
        # next, for each j for which w(j, s) = 1, we only want to add M[op][m][s][y] once to g[op][m][j][s], not twice.
        self.column_index: list[set[int]] = [set() for _ in range(1 << constraint_system.l_words)]
        for i, (I_a, I_b, I_c) in enumerate(constraint_system.and_constraints):
            for j, op, s in I_a:
                self.a[i] ^= op(self.witness[j], s)
                self.column_index[j].add((op, "a", s))
            for j, op, s in I_b:
                self.b[i] ^= op(self.witness[j], s)
                self.column_index[j].add((op, "b", s))
            for j, op, s in I_c:
                self.c[i] ^= op(self.witness[j], s)
                self.column_index[j].add((op, "c", s))

        assert all(self.a[i] & self.b[i] == self.c[i] for i in range(1 << constraint_system.l_and))

    def _directly_evaluate(self, r_zhat_prime, r_x_prime):
        # directly computes and coughs up the values of â(Ẑ, X), b̂(Ẑ, X) and ĉ(Ẑ, X) on given inputs r'_ẑ, r'_x.
        # verifier can use the resulting things as the claims α_a, α_b, α_c to feed into the shift reduction
        # in practice, these values will probably "pop out" for free at the end of the prover's univariate skip.
        # as for the verifier, my guess is that the prover will nondeterministically send them to the verifier,
        # so that the verifier can wrap up the univariate skip. pending the correctness of the supplied values.
        # the verifier will thus be done with univariate skip, and can turn his attention to the shift reduction.

        # this method can be viewed as a stand-in for the prover's univariate skip, in terms of the state we're left in.
        # will take this opportunity to memoize l_tilde ≔ L̃(r'_ẑ, i); the prover would compute this at the end of skip
        assert len(r_x_prime) == self.constraint_system.l_and
        self.l_tilde = univariate_expand(self.field, r_zhat_prime)
        self.r_x_prime = r_x_prime

        def extrapolate_bitstring(value):
            # interpreting `value` as the (bit-valued!) evaluations of some univariate polynomial over the domain D,
            # returns that polynomial's value at r'_x. because we precomputed L̃(r'_ẑ, i), just now need a subset sum.
            return sum((self.l_tilde[i] for i in range(64) if value >> i & 1), self.field.zero())

        def evaluate(m):
            univariate = [extrapolate_bitstring(m[i]) for i in range(1 << self.constraint_system.l_and)]  # specialize
            for i in range(self.constraint_system.l_and):  # now fold
                for j in range(1 << self.constraint_system.l_and - i - 1):
                    univariate[j] = univariate[j << 1] + r_x_prime[i] * (univariate[j << 1 | 1] - univariate[j << 1])
            return univariate[0]

        return (evaluate(self.a), evaluate(self.b), evaluate(self.c))

    def initiate_first_phase(self, lambda_a, lambda_b, lambda_c):
        self.lambdas = {  # will use these in second phase
            "a": lambda_a,
            "b": lambda_b,
            "c": lambda_c,
        }
        h = {
            ShiftOp.SLL: [],
            ShiftOp.SRL: [],
            ShiftOp.SRA: [],
        }
        for j in range(64):
            h[ShiftOp.SLL].append(self.l_tilde[j:] + [self.field.zero()] * j)
            h[ShiftOp.SRL].append(list(reversed(self.l_tilde[: j + 1])) + [self.field.zero()] * (64 - 1 - j))
            h[ShiftOp.SRA].append(list(reversed(self.l_tilde[: j + 1])) + [self.field.zero()] * (64 - 1 - j))
        for s in range(1, 64):
            h[ShiftOp.SRA][63][s] += h[ShiftOp.SRA][63][s - 1]

        r_x_tensor = tensor_expand(self.field, self.r_x_prime, self.constraint_system.l_and)
        # for each op × m × s, M[op][m][s] is Mₒₚ(r'ₓ, Y, s), the partial specialization of Mₒₚ(X, Y, s) at r'ₓ.
        # this looks like taking the Mₒₚ(X, Y, s) matrix, which is n_{and} × n_{word}, and folding it by ⊗(r'ₓ).
        # i.e., we should tensor r'ₓ, and interpret the result as a row-vector, and multiply Mₒₚ on the left.
        # the result is a length-n_{words} row-vector; moreover, we have a such thing for each op × m × s.
        # since Mₒₚ(X, Y, s) is sparse + binary, we can get this by allocating n_{words} and iterating through the 1s.
        self.M = {  # store these. will reuse them in the second phase
            ShiftOp.SLL: {
                "a": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
                "b": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
                "c": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
            },
            ShiftOp.SRL: {
                "a": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
                "b": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
                "c": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
            },
            ShiftOp.SRA: {
                "a": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
                "b": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
                "c": [[self.field.zero()] * (1 << self.constraint_system.l_words) for _ in range(64)],
            },
        }  # outer index s; inner index y
        for x, (I_a, I_b, I_c) in enumerate(self.constraint_system.and_constraints):
            for y, op, s in I_a:
                self.M[op]["a"][s][y] += r_x_tensor[x]
            for y, op, s in I_b:
                self.M[op]["b"][s][y] += r_x_tensor[x]
            for y, op, s in I_c:
                self.M[op]["c"][s][y] += r_x_tensor[x]

        g = {  # outer index j, inner index s
            ShiftOp.SLL: {
                "a": [[self.field.zero()] * 64 for _ in range(64)],
                "b": [[self.field.zero()] * 64 for _ in range(64)],
                "c": [[self.field.zero()] * 64 for _ in range(64)],
            },
            ShiftOp.SRL: {
                "a": [[self.field.zero()] * 64 for _ in range(64)],
                "b": [[self.field.zero()] * 64 for _ in range(64)],
                "c": [[self.field.zero()] * 64 for _ in range(64)],
            },
            ShiftOp.SRA: {
                "a": [[self.field.zero()] * 64 for _ in range(64)],
                "b": [[self.field.zero()] * 64 for _ in range(64)],
                "c": [[self.field.zero()] * 64 for _ in range(64)],
            },
        }
        for y in range(1 << self.constraint_system.l_words):
            w_y = self.witness[y]
            for j in range(64):
                # todo: there is a bitwise / simd-ish vibe here; revisit whether there is some clever memory stuff
                if w_y & 1:  # we only want to read the witness once. this seems to be the way to do it
                    for op, m, s in self.column_index[y]:
                        g[op][m][j][s] += self.M[op][m][s][y]
                        # the alternative is to simply loop over all op, m, and s, and do the above line for every one.
                        # that would be perfectly correct, since when `self.column_index[y] = []`, M[op][m][s][y] = 0.
                        # it would also be more of a regular computation (simd stuff possible? not sure.).
                        # but all signs point to that being a massive number of adds, enough to make it much slower.
                        # heuristically (depending on the constraint system), this should save on adds by 64x at least.
                w_y >>= 1

        # now sum over m, consolidate all g's into the "a" accumulator (just as a place to keep them, arbitrary choice)
        for j in range(64):
            for s in range(64):
                g[ShiftOp.SLL]["a"][j][s] *= self.lambdas["a"]
                g[ShiftOp.SLL]["a"][j][s] += self.lambdas["b"] * g[ShiftOp.SLL]["b"][j][s]
                g[ShiftOp.SLL]["a"][j][s] += self.lambdas["c"] * g[ShiftOp.SLL]["c"][j][s]

                g[ShiftOp.SRL]["a"][j][s] *= self.lambdas["a"]
                g[ShiftOp.SRL]["a"][j][s] += self.lambdas["b"] * g[ShiftOp.SRL]["b"][j][s]
                g[ShiftOp.SRL]["a"][j][s] += self.lambdas["c"] * g[ShiftOp.SRL]["c"][j][s]

                g[ShiftOp.SRA]["a"][j][s] *= self.lambdas["a"]
                g[ShiftOp.SRA]["a"][j][s] += self.lambdas["b"] * g[ShiftOp.SRA]["b"][j][s]
                g[ShiftOp.SRA]["a"][j][s] += self.lambdas["c"] * g[ShiftOp.SRA]["c"][j][s]

        # NOW going to flatten all 3 h's and all 3 (summed) g's. this is purely a matter of memory layout.
        # in a real implementation, we should probably arrange things so that they start and stay flat the whole time,
        # i.e. as length-2¹² arrays the entire time. doing that would make the indexing above a bit more of a pain.
        # i.e., everywhere you see [j][s], that would need to be replaced with j << 6 | s, etc. etc.
        # note that we are doing our sumcheck in such a way that s is less-significant and j is more-significant.
        # j is the outer (row) index; s is the inner (column) index. summing all rows is concat in row-major order.
        multilinears = [
            sum(h[ShiftOp.SLL], []),
            sum(g[ShiftOp.SLL]["a"], []),
            sum(h[ShiftOp.SRL], []),
            sum(g[ShiftOp.SRL]["a"], []),
            sum(h[ShiftOp.SRA], []),
            sum(g[ShiftOp.SRA]["a"], []),
        ]
        composition = Polynomial(  # ∑ₒₚ hₒₚ ⋅ gₒₚ
            6,
            {
                (1, 1, 0, 0, 0, 0): self.field.one(),
                (0, 0, 1, 1, 0, 0): self.field.one(),
                (0, 0, 0, 0, 1, 1): self.field.one(),
            },
        )
        self.first_phase_sumcheck = Sumcheck(multilinears, composition)

    def initiate_second_phase(self):
        # assert: self.first_phase_sumcheck.round == 12
        r_s, r_j = self.first_phase_sumcheck.challenges[:6], self.first_phase_sumcheck.challenges[6:]
        r_j_tensor = tensor_expand(self.field, r_j, 6)
        w_r_j = [
            sum(
                (r_j_tensor[j] if self.witness[y] >> j & 1 else self.field.zero() for j in range(64)), self.field.zero()
            )
            for y in range(1 << self.constraint_system.l_words)
        ]

        # begin computing second multilinear of second phase of sumcheck. see note.
        # there are a lot of implementation choices here. various ways to proceed.
        # should we keep the respective vectors M̃ₘ,ₒₚ(r'ₓ, Y, rₛ) (for Y varying)---for all 9 (m, op)---separate,
        # and run separate sumchecks on them, batching the round polynomials out of the hot loop by λₘ and hₒₚ(rⱼ, rₛ)?
        # should we do some intermediate thing, where we pre-consolidate ∑ₘ λₘ ⋅ M̃ₘ,ₒₚ(r'ₓ, Y, rₛ), for each op,
        # and then batch the resulting three things using the coefficients hₒₚ(rⱼ, rₛ) outside of the hot loop?
        # my sense is to go full bore and combine all 9 things into a single table of n_words values, before proceeding.
        # the rough heuristic is that the M̃ₘ,ₒₚ(r'ₓ, Y, rₛ)s are fairly sparse: for each y, M̃ₘ,ₒₚ(r'ₓ, Y, rₛ) ≠ 0 iff
        # w[y] appears in some constraint for (m, op) (for SOME constraint index x and SOME shift amount s).
        # heuristically, this will only hold for let's say 1 out of the 9 possible (m, op)s. thus we can batch all 9
        # of these things into 1, in such a way that exploits sparsity---since the sumcheck algorithm sure-as-hell won't

        h_op_r_j_r_s = {
            ShiftOp.SLL: self.first_phase_sumcheck.multilinears[0][0],
            ShiftOp.SRL: self.first_phase_sumcheck.multilinears[2][0],
            ShiftOp.SRA: self.first_phase_sumcheck.multilinears[4][0],
        }

        r_s_tensor = tensor_expand(self.field, r_s, 6)
        M = [self.field.zero()] * (1 << self.constraint_system.l_words)
        for y in range(1 << self.constraint_system.l_words):
            for op, m, s in self.column_index[y]:
                M[y] += h_op_r_j_r_s[op] * self.lambdas[m] * r_s_tensor[s] * self.M[op][m][s][y]

        multilinears = [w_r_j, M]
        composition = Polynomial(2, {(1, 1): self.field.one()})
        self.second_phase_sumcheck = Sumcheck(multilinears, composition)
