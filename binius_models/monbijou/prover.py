from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem
from ..finite_fields.utils import tensor_expand
from ..ips.polynomial import Polynomial
from ..ips.sumcheck import Sumcheck
from .constraint_system import ConstraintSystem, ShiftOp
from .utils import univariate_expand

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

        # in one pass over the constraint system, both accumulate our a, b, c values and set up the "column index".
        # the a, b, c are just as in the paper, and are computed in the same way as there (by shifting and accumulating)
        # the column index is an auxiliary data structure which will be used to speed up the sumcheck's first phase.
        # the point is to store, for each value y ∈ {0, … , n_{words} – 1} indexing into the witness, a dictionary,
        # whose keys are those (op, m, s)s for which the yᵗʰ column of M_{op, m}(X, y, s) is not identically zero.
        # the VALUE at self.column_index[y][(op, m, s)] will ultimately be Mₒₚ(r'ₓ, Y, s). see below for computation.
        # these will be exactly the (op, m, s) for which the prover needs to update the "gacc" accumulator; see below.
        self.column_index: list[dict[tuple[ShiftOp, str, int], F]] = [{} for _ in range(1 << constraint_system.l_words)]
        for x, (I_a, I_b, I_c) in enumerate(constraint_system.and_constraints):
            for y, op, s in I_a:
                self.a[x] ^= op(self.witness[y], s)
                self.column_index[y][(op, "a", s)] = self.field.zero()
            for y, op, s in I_b:
                self.b[x] ^= op(self.witness[y], s)
                self.column_index[y][(op, "b", s)] = self.field.zero()
            for y, op, s in I_c:
                self.c[x] ^= op(self.witness[y], s)
                self.column_index[y][(op, "c", s)] = self.field.zero()
        # note: for each y, the KEYS (m, op, s) of column_index[y] will only depend on the constraint system, while
        # the VALUES will ultimately depend on quantities unknown till runtime, namely r'ₓ (though not on the witness!).
        # thus these need to be purged if you ever want to run the same prover twice, even if the witness doesn't change
        # this is an odd thing to do, so i will let them sit (i.e., don't bother purging at the beginning of proofgen).
        # technically, the constraint-system-dependent part, the witness-dependent part, and the runtime-dependent part
        # could / should all be broken apart, for exactness. i don't know that it's necessary to do that for now?

        assert all(self.a[i] & self.b[i] == self.c[i] for i in range(1 << constraint_system.l_and))

    def _directly_evaluate(self, r_zhat_prime, r_x_prime):
        # directly computes and coughs up the values of â(Ẑ, X), b̂(Ẑ, X) and ĉ(Ẑ, X) on given inputs r'_ẑ, r'_x.
        # verifier can use the resulting things as the claims α_a, α_b, α_c to feed into the shift reduction.
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
            # returns that polynomial's value at r'_ẑ. because we precomputed L̃(r'_ẑ, i), just now need a subset sum.
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
        h: dict[ShiftOp, list[list[F]]] = {
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

        r_x_prime_tensor = tensor_expand(self.field, self.r_x_prime, self.constraint_system.l_and)
        # for each y ∈ {0, … , n_{words} – 1}, for each op × m × s for which Mₒₚ(X, y, s) is not completely empty,
        # we are going to accumulate Mₒₚ(r'ₓ, Y, s) into column_index[y][(op, m, s)]. since nonempty, it will be nonzero
        # to do this we just need to add ⊗(r'ₓ)[x] to column_index[y][(op, m, s)] for each x for which Mₒₚ(x, y, s) = 1.
        for x, (I_a, I_b, I_c) in enumerate(self.constraint_system.and_constraints):
            for y, op, s in I_a:
                self.column_index[y][(op, "a", s)] += r_x_prime_tensor[x]
            for y, op, s in I_b:
                self.column_index[y][(op, "b", s)] += r_x_prime_tensor[x]
            for y, op, s in I_c:
                self.column_index[y][(op, "c", s)] += r_x_prime_tensor[x]

        # these g's are actually surprisingly large. we have 9 × 64 × 64 field elements to store; this is ≈≈ 576 KiB.
        # this is about as much memory as we are liable to need in the second phase, for n_{words} == 15, so it seems ok
        # and moreover, we will do better than this _at best_ by a factor of 3, in view of how the sumcheck below goes.
        # optimizing this seems not simple, and not sure if we need to go down that path or not.
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
                    # note that y, and so likewise the items ((op, m, s), value) below, stay constant even as j varies.
                    # thus the innermost loop down below and the loop over j above could be interchanged.
                    # for each j, we are adding the same thing to the same places if w[y] >> j == 1, just with j changed
                    # the written approach has the benefit that we only need to read the witness exactly once
                    for (op, m, s), value in self.column_index[y].items():
                        g[op][m][j][s] += value
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
        multilinears: list[list[F]] = [
            sum(h[ShiftOp.SLL], []),
            sum(g[ShiftOp.SLL]["a"], []),
            sum(h[ShiftOp.SRL], []),
            sum(g[ShiftOp.SRL]["a"], []),
            sum(h[ShiftOp.SRA], []),
            sum(g[ShiftOp.SRA]["a"], []),
        ]
        composition = Polynomial(  # ∑ₒₚ hₒₚ ⋅ gₒₚ
            self.field,
            6,
            {
                (1, 1, 0, 0, 0, 0): self.field.one(),
                (0, 0, 1, 1, 0, 0): self.field.one(),
                (0, 0, 0, 0, 1, 1): self.field.one(),
            },
        )
        self.first_phase_sumcheck = Sumcheck(self.field, multilinears, composition)

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
            for (op, m, s), value in self.column_index[y].items():
                M[y] += h_op_r_j_r_s[op] * self.lambdas[m] * r_s_tensor[s] * value
                # possible savings above: if we could "group" the above list items by op, then could factor out h_op.
                # not sure how significant this would be---will there be one witness item w[y] and one `op`, for which
                # op(w[y], s) shows up in a constraint for many different s's and multiple (say, more than one) m's?
                # i.e., will you shift the same word in the same direction, but for many different shift amounts?
                # if so, then this might be worth it---wouldn't be hard to re-express the column index this way.
                # seems a bit implausible, but who knows---and the effort of doing this optimization would be quite low.
                # instead of a flat list of (op, m, s)s per y, we would have 3 lists, grouped by `op`. revisit this!

        multilinears = [w_r_j, M]
        composition = Polynomial(self.field, 2, {(1, 1): self.field.one()})
        self.second_phase_sumcheck = Sumcheck(self.field, multilinears, composition)
