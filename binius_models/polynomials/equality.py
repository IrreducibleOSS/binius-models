from typing import TypeVar

from binius_models.finite_fields.finite_field import FiniteFieldElem

F = TypeVar("F", bound=FiniteFieldElem)


def eq(field: type[F], x: F, y: F) -> F:
    """
    Evaluation of the multilinear polynomial which indicates the condition x == y.
    """
    return x * y + (field.one() - x) * (field.one() - y)


class EqualityIndicator:
    def __init__(self, field: type[F], v: int) -> None:
        """Constructs an equality indicator polynomial.

        :param field: the field
        :param v: number of variables
        """
        self.field = field
        self.v = v

    def evaluate_at_point(self, x: list[F], y: list[F]) -> F:
        """Evaluates the equality indicator polynomial at a point."""
        # O(ν)-time alg
        assert len(x) == self.v
        assert len(y) == self.v
        value = self.field.one()
        for k in range(self.v):
            value *= eq(self.field, x[k], y[k])
        return value

    def evaluate_over_hypercube(self, r: list[F]) -> list[F]:
        """Evaluates the equality indicator polynomial over the entire hypecube."""
        array = [self.field.one()] * (1 << len(r))
        for k in range(len(r)):
            for i in range(1 << k):
                array[1 << k | i] = array[i] * r[k]
                array[i] -= array[1 << k | i]
        return array


def partially_evaluate_multilinear_extension(indicator: EqualityIndicator, f: list[F], r: list[F]) -> list[F]:
    # given a _partial_ input point vector of length ≤ v, returns the array of partial evaluations
    assert len(f) == 1 << indicator.v
    assert len(r) in range(indicator.v + 1)  # redundant; will happen inside evaluate_over-hypercube
    array = indicator.evaluate_over_hypercube(r)
    b = indicator.v - len(r)
    return [sum((f[j << b | i] * array[j] for j in range(1 << len(r))), indicator.field.zero()) for i in range(1 << b)]


def evaluate_multilinear_extension(indicator: EqualityIndicator, f: list[F], r: list[F]) -> F:
    assert len(f) == 1 << indicator.v
    assert len(r) == indicator.v  # redundant; will happen inside evaluate_over-hypercube
    array = indicator.evaluate_over_hypercube(r)
    return sum((f[i] * array[i] for i in range(1 << indicator.v)), indicator.field.zero())
