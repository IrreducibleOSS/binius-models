from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem

F = TypeVar("F", bound=FiniteFieldElem)


class Polynomial(Generic[F]):
    def __init__(self, field: type[F], variables: int, data: dict[tuple[int, ...], F]) -> None:
        self.field = field
        self.variables = variables
        self.degree = 0  # `degree` refers to the _total degree_ (!) of the multivariate polynomial.
        for multidegree, coefficient in data.items():
            assert len(multidegree) == variables  # each key is a multi-degree of length `variables`
            assert all(degree >= 0 for degree in multidegree)  # all exponents are nonnegative
            assert coefficient  # pointless to have 0 coefficients; let's just exclude
            self.degree = max(self.degree, sum(multidegree))
        self.data = data

    def evaluate(self, argument: list[F]) -> F:
        assert len(argument) == self.variables, f"arguments: {len(argument)}, variables: {self.variables}"
        result = self.field.zero()
        for multidegree, coefficient in self.data.items():
            monomial = self.field.one()
            for i, degree in enumerate(multidegree):
                monomial *= argument[i] ** degree
            result += coefficient * monomial
        return result
