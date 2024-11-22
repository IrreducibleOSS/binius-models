# (C) 2024 Irreducible Inc.
from __future__ import annotations

from math import comb
from random import randint, shuffle
from typing import Type, TypeVar

from sympy import Expr, Pow, S, symbols

from binius_models.finite_fields.tower import BinaryTowerFieldElem

E = TypeVar("E", bound=BinaryTowerFieldElem)


class Polynomial:
    """This class supercedes sumcheck.Polynomial128.
    TODO Replace Polynomial128 all over the place.
    """

    def __init__(self, elem_t: Type[E], variables: int, terms: dict[tuple[int, ...], int]) -> None:
        self.elem_t = elem_t
        self.variables = variables

        for multidegree, coefficient in terms.items():
            assert len(multidegree) == variables
            assert all(degree >= 0 for degree in multidegree)
            assert coefficient > 0, "Pointless to have 0 coefficients; let's just exclude"

        self.degree = max(sum(key) for key in terms.keys())
        self.terms = {multidegree: self.elem_t(coefficient) for multidegree, coefficient in terms.items()}
        self.max_terms = Polynomial.get_max_terms(self.degree, variables)

    def evaluate(self, argument: list[E]) -> E:
        assert len(argument) == self.variables
        result = self.elem_t.zero()
        for multidegree, coefficient in self.terms.items():
            non_zero_indices = [index for index, degree in enumerate(multidegree) if degree > 0]
            monomial = self.elem_t.one()
            for index in non_zero_indices:
                monomial *= argument[index] ** multidegree[index]
            result += coefficient * monomial
        return result

    def to_symbols(self) -> Expr:
        symbol_dict = {f"x{i}": symbols(f"x{i}") for i in range(self.variables)}

        symbolic: Expr = S(0)

        for md, coeff in self.terms.items():
            term = S(coeff.value)
            for i, deg in enumerate(md):
                term *= Pow(symbol_dict[f"x{i}"], deg)
            symbolic += term

        return symbolic

    def __str__(self) -> str:
        return str(self.to_symbols())

    @staticmethod
    def get_max_terms(
        degree: int,
        variables: int,
    ) -> int:
        return comb(degree + variables, variables)

    @staticmethod
    def random(elem_t: Type[E], degree: int, variables: int, term_num: int | None = None) -> Polynomial:
        if term_num is None:
            term_num = Polynomial.get_max_terms(degree, variables)
        else:
            assert 0 < term_num <= Polynomial.get_max_terms(degree, variables), "Invalid term_num"

        def random_multidegree() -> tuple[int, ...]:
            multidegree = [0 for _ in range(variables)]

            remaining_degree = randint(0, degree)
            for i in range(variables - 1):
                degree_ = randint(0, remaining_degree)
                remaining_degree -= degree_
                multidegree[i] = degree_
            multidegree[-1] = remaining_degree

            shuffle(multidegree)
            assert sum(multidegree) <= degree
            return tuple(multidegree)

        multidegrees: set[tuple[int, ...]] = set()
        while len(multidegrees) < term_num:
            multidegrees.add(random_multidegree())

        multidegrees_: list[tuple[int, ...]] = list(multidegrees)
        if max(sum(md) for md in multidegrees) != degree:
            # Ensure that the degree of the polynomial is exactly `degree`
            diff = degree - sum(multidegrees_[0])
            multidegrees_[0] = multidegrees_[0][0] + diff, *multidegrees_[0][1:]

        def non_zero_random() -> int:
            result = 0
            while result == 0:
                result = elem_t.random().value
            return result

        return Polynomial(elem_t, variables, {md: non_zero_random() for md in sorted(multidegrees_)})
