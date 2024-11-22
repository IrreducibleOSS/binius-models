import copy

from ..finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField


class Elem1b(BinaryTowerFieldElem):
    field = FanPaarTowerField(0)


class Elem8b(BinaryTowerFieldElem):
    field = FanPaarTowerField(3)


class Elem128b(BinaryTowerFieldElem):
    field = FanPaarTowerField(7)


def compute_switchover(v: int, d: int, ratio: int) -> int:
    # returns: at which point should we switch from b × e muls, plus tensor expansion, to classical-style folding?
    # parameter `ratio`: what is the cost of an e × e mult, relative to that of a b × e?

    # in the simplest case len(multilinears) == 1 and subfield size 1 bit (and ignoring XORs), the best is at v/2.
    # more generally, allowing len(multilinears) == d you can calculate that the breakeven point happens at exactly
    # (log(d) + v) / 2. why? for each round, the cost of self.receive_challenge is:
    # - if round < switchover: exactly 2ʳᵒᵘⁿᵈ e × e mults.
    # - if round ≥ switchover, exactly d ⋅ 2ᵛ⁻ʳᵒᵘⁿᵈ e × e mults.
    # thus we're interested in the smallest r for which d ⋅ 2ᵛ⁻ʳ ≤ 2ʳ first becomes true.
    # taking logs of both sides, we see the inequality log(d) + v ≤ 2 ⋅ r, and we're done.

    # it becomes more complicated if we want to count the cost of b × e mults---
    # as we might, say, as soon as the field size of the polys is > 1 bit, or if we care about the cost of XORs.
    # of course the higher the cost of b × es, the earlier we will want the switchover to actually happen.
    # indeed, now counting b × es, the total cost of each round is 2ᵛ⁻ʳᵒᵘⁿᵈ⁻¹ ⋅ deg polynomial evaluations,
    # where write `deg` for the total degree of the composition polynomial (it equals d when it's just a product),
    # (this cost is there regardless of pre-switchover or not), PLUS, in addition:
    # - if round < switchover: d ⋅ 2ᵛ b × e (compute folds on the fly) + 2ʳᵒᵘⁿᵈ e × e (tensor expand)
    # - if round > switchover: d ⋅ 2ᵛ⁻ʳᵒᵘⁿᵈ e × e (classical folding)
    # - if round == switchover: d ⋅ 2ᵛ b × e + d ⋅ 2ᵛ⁻ʳᵒᵘⁿᵈ e × e (on-the-fly and fold).
    # when you work it out, for arbitrary switchover r ∈ {0, ..., v - 1}, the total cost across all rounds becomes:
    # f(r) :=
    # (2ʳ − 1) e × e (total work of tensor expansion, up to but excluding the switchover round)
    # + d ⋅ r ⋅ 2ᵛ b × e (total work of only-the-fly folds, including switchover round, but excluding 0th round)
    # + d ⋅ (2ᵛ⁻ʳ⁺¹ - 2) e × e (total work of classical folds, including the switchover round and following rounds)
    # so our goal is to choose r ∈ {0, ..., v - 1} so as to minimze f(r), which is a straightforward calculation.
    # though the minimal value in practice will depend on the relative cost of b × es versus e × es, as well as d.
    # you can see that we have an amount of b × es which increases linearly in r (penalty for delaying the switch),
    # as well as the familiar exponentially increasing and decreasing costs in e × es of tensor and folding, resp.
    return min(range(v), key=lambda r: ((1 << r) - 1) * ratio + d * r * (1 << v) + d * ((1 << v - r + 1) - 2) * ratio)


# Corresponds to `binius_core::polynomial::extrapolate_line()`.
def linearly_interpolate(points: tuple[Elem128b, Elem128b], r: Elem128b) -> Elem128b:
    return points[0] + (points[1] - points[0]) * r


class Polynomial128:
    def __init__(self, variables: int, data: dict[tuple[int, ...], Elem128b]) -> None:
        self.variables = variables
        self.degree = 0  # `degree` refers to the _total degree_ (!) of the multivariate polynomial.
        for multidegree, coefficient in data.items():
            assert len(multidegree) == variables  # each key is a multi-degree of length `variables`
            assert all(degree >= 0 for degree in multidegree)  # all exponents are nonnegative
            assert coefficient  # pointless to have 0 coefficients; let's just exclude
            self.degree = max(self.degree, sum(multidegree))
        self.data = data

    def evaluate(self, argument: list[Elem128b]) -> Elem128b:
        assert len(argument) == self.variables, f"arguments: {len(argument)}, variables: {self.variables}"
        result = Elem128b.zero()
        for multidegree, coefficient in self.data.items():
            monomial = Elem128b.one()
            for i, degree in enumerate(multidegree):
                monomial *= argument[i] ** degree
            result += coefficient * monomial
        return result


def mul_matrix_vec(m: list[list[Elem128b]], v: list[Elem128b]) -> list[Elem128b]:
    res = []
    for i in range(len(m)):
        p = Elem128b.zero()
        for j in range(len(v)):
            p += m[i][j] * v[j]
        res.append(p)
    return res


# Corresponds to `binius_core::polynomial::evaluate_univariate()`.
def evaluate_univariate(coeffs: list[Elem128b], x: Elem128b) -> Elem128b:
    assert coeffs
    eval = coeffs[-1]
    for coeff in reversed(coeffs[:-1]):
        eval = eval * x + coeff
    return eval


def _identity_matrix(n: int) -> list[list[Elem128b]]:
    return [[Elem128b.one() if i == j else Elem128b.zero() for j in range(n)] for i in range(n)]


# Corresponds to `binius_core::linalg::Matrix::inverse_into()`.
def inverse_matrix(m: list[list[Elem128b]]) -> list[list[Elem128b]]:
    n = len(m)
    assert len(m[0]) == n
    tmp = copy.deepcopy(m)
    out = _identity_matrix(n)
    for i in range(n):
        pivot = i
        while pivot < n and tmp[pivot][i] == Elem128b.zero():
            pivot += 1
        assert pivot < n
        if pivot != i:
            tmp[i], tmp[pivot] = tmp[pivot], tmp[i]
            out[i], out[pivot] = out[pivot], out[i]

        scalar = tmp[i][i].inverse()
        for j in range(n):
            tmp[i][j] *= scalar
            out[i][j] *= scalar

        for j in range(n):
            if j != i:
                scalar = tmp[j][i]
                for k in range(n):
                    tmp[j][k] -= tmp[i][k] * scalar
                    out[j][k] -= out[i][k] * scalar

    assert tmp == _identity_matrix(n)
    return out


# Corresponds to `binius_core::polynomial::vandermonde()`.
def vandermonde(n: int) -> list[list[Elem128b]]:
    vandermonde = []
    for i in range(n):
        product = Elem128b.one()
        row = [product]
        for j in range(1, n):
            product *= Elem128b(i)
            row.append(product)
        vandermonde.append(row)
    return vandermonde


# Corresponds to `binius_core::polynomial::util::tensor_prod_eq_ind()`.
def _tensor_prod_eq_ind(log_n_values: int, values: list[Elem128b], extra_query_coordinates: list[Elem128b]):
    for i, r_i in enumerate(extra_query_coordinates):
        prev_length = 2 ** (log_n_values + i)
        assert prev_length * 2 <= len(values)
        for j in range(prev_length):
            prod = values[j] * r_i
            values[j] -= prod
            values[prev_length + j] = prod
    return values


# Corresponds to `binius_core::polynomial::transparent::eq_ind::EqIndPartialEval::new().multilinear_extension()`.
def multilinear_query(r: list[Elem128b]):
    expanded_query = [Elem128b.one()]
    expanded_query.extend([Elem128b.zero() for _i in range((2 ** len(r)) - 1)])
    return _tensor_prod_eq_ind(0, expanded_query, r)


def test_query():
    res = multilinear_query([])
    assert res == [Elem128b(x) for x in [1]]
    res = multilinear_query([Elem128b(x) for x in [2]])
    assert res == [Elem128b(x) for x in [3, 2]]
    res = multilinear_query([Elem128b(x) for x in [2, 2]])
    assert res == [Elem128b(x) for x in [2, 1, 1, 3]]
    res = multilinear_query([Elem128b(x) for x in [2, 2, 2]])
    assert res == [Elem128b(x) for x in [1, 3, 3, 2, 3, 2, 2, 1]]
    res = multilinear_query([Elem128b(x) for x in [2, 2, 2, 2]])
    assert res == [Elem128b(x) for x in [3, 2, 2, 1, 2, 1, 1, 3, 2, 1, 1, 3, 1, 3, 3, 2]]


def test_evaluate_univariate():
    for coeffs, challenge, output in [
        ([143, 234, 184, 110], 10, 45),
        ([40, 25, 156, 90], 12, 147),
        ([50, 90, 249, 53], 14, 113),
        ([75, 234, 184, 110], 18, 191),
    ]:
        coeffs = [Elem128b(x) for x in coeffs]
        challenge = Elem128b(challenge)
        output = Elem128b(output)
        assert output == evaluate_univariate(coeffs, challenge)


def test_inverse_vandermonde():
    m = inverse_matrix(vandermonde(4))
    assert m[0] == [Elem128b(x) for x in [1, 0, 0, 0]]
    assert m[1] == [Elem128b(x) for x in [0, 1, 3, 2]]
    assert m[2] == [Elem128b(x) for x in [0, 1, 2, 3]]
    assert m[3] == [Elem128b(x) for x in [1, 1, 1, 1]]
