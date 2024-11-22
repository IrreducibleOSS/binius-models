from typing import TypeVar

from binius_models.finite_fields.finite_field import FiniteFieldElem

from .equality import EqualityIndicator, eq, partially_evaluate_multilinear_extension

F = TypeVar("F", bound=FiniteFieldElem)


class ShiftIndicator:
    def __init__(self, field: type[F], v: int, b: int, o: int) -> None:
        """Constructs a shift indicator polynomial.

        :param field: the field
        :param v: number of variables
        :param b: log2 of block size
        :param o: shift offset
        """
        assert b in range(v + 1)
        assert o in range(1 << b)
        self.field = field
        self.v = v
        self.b = b
        self.o = o

    def evaluate_at_point(self, x: list[F], y: list[F]) -> F:
        """Evaluates the equality indicator polynomial at a point."""
        # O(b)-time alg
        assert len(x) == self.v
        assert len(y) == self.v
        s_ind_p = self.field.one()
        s_ind_pp = self.field.zero()
        for k in range(self.b):
            o_k = self.o >> k & 1
            if o_k:
                s_ind_p_new = (self.field.one() - x[k]) * y[k] * s_ind_p
                s_ind_pp_new = x[k] * (self.field.one() - y[k]) * s_ind_p + eq(self.field, x[k], y[k]) * s_ind_pp
            else:
                s_ind_p_new = eq(self.field, x[k], y[k]) * s_ind_p + (self.field.one() - x[k]) * y[k] * s_ind_pp
                s_ind_pp_new = x[k] * (self.field.one() - y[k]) * s_ind_pp
            # roll over results
            s_ind_p = s_ind_p_new
            s_ind_pp = s_ind_pp_new
        return s_ind_p + s_ind_pp

    def evaluate_over_hypercube(self, r: list[F]) -> list[F]:
        """Evaluates the equality indicator polynomial over the entire hypecube."""
        # we are going to compute { s-ind_{b, o}(u₀, … , u_{b - 1}, r₀, … , r_{b - 1}) | u ∈ ℬ_b }.
        # note that we only pay attention to the lowest b bits of r in this function.
        # total time is O(2ᵇ) 𝒯_τ-operations! this is optimal (in light of output size).
        # the result will be exactly the "vector" of elements we need to dot f̃'s evaluations with; see paper.
        assert len(r) == self.v
        self.r = r
        s_ind_p = [self.field.one()] * (1 << self.b)
        s_ind_pp = [self.field.zero()] * (1 << self.b)

        for k in range(self.b):
            o_k = self.o >> k & 1
            for i in range(1 << k):  # complexity: just two multiplications per iteration!
                if o_k:
                    s_ind_pp[1 << k | i] = s_ind_pp[i] * r[k]
                    s_ind_pp[i] -= s_ind_pp[1 << k | i]
                    s_ind_p[1 << k | i] = s_ind_p[i] * r[k]  # gimmick: use this as a stash slot
                    s_ind_pp[1 << k | i] += s_ind_p[i] - s_ind_p[1 << k | i]  # * 1 - r
                    s_ind_p[i] = s_ind_p[1 << k | i]  # now move to lower half
                    s_ind_p[1 << k | i] = self.field.zero()  # clear upper half
                else:
                    s_ind_p[1 << k | i] = s_ind_p[i] * r[k]
                    s_ind_p[i] -= s_ind_p[1 << k | i]
                    s_ind_pp[1 << k | i] = s_ind_pp[i] * (self.field.one() - r[k])
                    s_ind_p[i] += s_ind_pp[i] - s_ind_pp[1 << k | i]
                    s_ind_pp[i] = self.field.zero()  # clear lower half
        return [p + pp for p, pp in zip(s_ind_p, s_ind_pp)]


def evaluate_shift_polynomial(indicator: ShiftIndicator, f: list[F], r: list[F]) -> F:
    assert len(f) == 1 << indicator.v
    assert len(r) == indicator.v  # redundant; will happen inside evaluate_over-hypercube
    # this routine evaluates shift_{b, o}(f)(r), given f as input.
    # now uses the highest bits r_b, ..., r_{ν - 1}; the low b bits already got used above.
    # total time is O(2ᵛ) 𝒯_τ-operations. (this is optimal: look at the size of the input.)
    # we assume that f is given to us in the big-endian order;
    # that is, as the array [f(0, ...., 0), f(0, ...., 0, 1), ..., f(1, ..., 1)].

    # now we compute the result, which of course depends on f.
    # there is a slight subtlety that we are evaluating f̃ outside of the cube (i.e., at least if b < ν).
    # specifically, what we need throughout is the set { f̃( u₀, … , u_{b - 1}, r_b, … r_{ν - 1}) | u ∈ ℬ_b }.
    # the way to compute this is just some standard tensor-math, which we do now.
    array = indicator.evaluate_over_hypercube(r)
    auxiliary = EqualityIndicator(indicator.field, indicator.v)  # auxiliary indicator for partial tensor....!!!
    partials = partially_evaluate_multilinear_extension(auxiliary, f, r[indicator.b :])  # most-significant components!
    return sum((partials[i] * array[i] for i in range(1 << indicator.b)), indicator.field.zero())
