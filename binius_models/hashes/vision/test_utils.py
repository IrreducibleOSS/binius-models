# (C) 2024 Irreducible Inc.

from typing import TypeVar

from pytest import mark

from binius_models.finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField
from binius_models.hashes.vision.utils import (
    determinant,
    matrix_inverse,
    naive_determinant,
    naive_matrix_inverse,
    transpose,
)

F = TypeVar("F", bound="BinaryTowerFieldElem")


class Elem(BinaryTowerFieldElem):
    field = FanPaarTowerField(5)


@mark.parametrize("size", [3, 24])
def test_transpose(size: int) -> None:
    """Test the transpose function"""

    arange = [Elem(x) for x in range(size**2)]
    matrix = [[x for x in arange[i * size : (i + 1) * size]] for i in range(size)]

    assert transpose(transpose(matrix)) == matrix


@mark.parametrize("size", [4, 6])
def test_determinant(size: int) -> None:
    """Test the determinant function"""

    arange = [Elem.random() for _ in range(size**2)]
    matrix = [[x for x in arange[i * size : (i + 1) * size]] for i in range(size)]

    assert naive_determinant(matrix) == determinant(matrix)


@mark.parametrize("size", [4, 6])
def test_matrix_inverse(size: int) -> None:
    """Test the matrix_inverse function"""

    arange = [Elem.random() for _ in range(size**2)]
    matrix = [[x for x in arange[i * size : (i + 1) * size]] for i in range(size)]

    assert naive_matrix_inverse(matrix) == matrix_inverse(matrix)
