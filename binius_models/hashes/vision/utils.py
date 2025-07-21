# (C) 2024 Irreducible Inc.

from dataclasses import dataclass
from hashlib import shake_256
from itertools import combinations
from math import ceil, log2
from typing import Any, Generic, TypeVar

from binius_models.finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField

F = TypeVar("F", bound="BinaryTowerFieldElem")


def naive_determinant(matrix: list[list[F]]) -> F:
    """Calculate the determinant of a square matrix"""

    n = len(matrix)
    zero = matrix[0][0].zero()

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = zero
    for c in range(n):
        # remove ((-1)**c) because we are in a binary field
        det += matrix[0][c] * naive_determinant(matrix_minor(matrix, 0, c))
    return det


def matrix_from_circulant(circulant: list[F]) -> list[list[F]]:
    """Return a matrix from a circulant vector"""
    n = len(circulant)
    matrix = [[circulant[(j - i) % n] for j in range(n)] for i in range(n)]
    return matrix


def naive_matrix_inverse(matrix: list[list[F]]) -> list[list[F]]:
    det = determinant(matrix)
    if det == matrix[0][0].zero():
        raise ValueError("Matrix is not invertible")
    det_inv = det.inverse()

    adj = adjugate(matrix)
    return [[adj[r][c] * det_inv for c in range(len(adj))] for r in range(len(adj))]


def determinant(matrix: list[list[F]]) -> F:
    one = matrix[0][0].one()
    zero = matrix[0][0].zero()
    n = len(matrix)
    try:
        _, U = _lu_decomposition(matrix)
        det = one
        for i in range(n):
            det *= U[i][i]  # Product of diagonal elements of U
    except ValueError:
        det = zero
    return det


def matrix_inverse(matrix: list[list[F]]) -> list[list[F]]:
    one = matrix[0][0].one()
    zero = matrix[0][0].zero()
    n = len(matrix)

    L, U = _lu_decomposition(matrix)

    inverse = [[zero for _ in range(n)] for _ in range(n)]

    for i in range(n):
        e = [one if i == j else zero for j in range(n)]
        y = _forward_substitution(L, e)
        x = _backward_substitution(U, y)
        inverse[i] = x

    return transpose(inverse)


def _lu_decomposition(A: list[list[F]]) -> tuple[list[list[F]], list[list[F]]]:
    """
    Perform LU decomposition of a square matrix A.
    A is decomposed into L and U such that A = LU.
    L is a lower triangular matrix and U is an upper triangular matrix.
    """
    n = len(A)
    zero = A[0][0].zero()
    one = A[0][0].one()

    L = [[zero for _ in range(n)] for _ in range(n)]  # Lower
    U = [[zero for _ in range(n)] for _ in range(n)]  # Upper

    for i in range(n):
        # Upper Triangular (U)
        for k in range(i, n):
            sum = zero
            for j in range(i):
                sum += L[i][j] * U[j][k]
            U[i][k] = A[i][k] - sum

        # Lower Triangular (L)
        for k in range(i, n):
            if i == k:
                L[i][i] = one  # Diagonal as 1
                if U[i][i] == zero:  # Matrix is singular
                    raise ValueError("Matrix is not invertible")
            else:
                sum = zero
                for j in range(i):
                    sum += L[k][j] * U[j][i]
                L[k][i] = (A[k][i] - sum) * U[i][i].inverse()

    return L, U


def _forward_substitution(L: list[list[F]], B: list[F]) -> list[F]:
    n = len(L)
    zero = L[0][0].zero()
    Y = [zero for _ in range(n)]
    for i in range(n):
        Y[i] = B[i] - sum([L[i][j] * Y[j] for j in range(i)], start=zero)
    return Y


def _backward_substitution(H: list[list[F]], Y: list[F]) -> list[F]:
    n = len(H)
    zero = H[0][0].zero()
    X = [zero for _ in range(n)]

    for i in range(n - 1, -1, -1):
        X[i] = (Y[i] - sum([H[i][j] * X[j] for j in range(i + 1, n)], start=zero)) * H[i][i].inverse()
    return X


def matrix_minor(matrix: list[list[F]], i: int, j: int) -> list[list[F]]:
    """Return the minor of matrix[i][j]"""
    return [row[:j] + row[j + 1 :] for row in (matrix[:i] + matrix[i + 1 :])]


def adjugate(matrix: list[list[F]]) -> list[list[F]]:
    n = len(matrix)
    cofactors: list[list[F]] = [[matrix[0][0].zero() for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            # removing ((-1) ** (r + c)) * because we are in a binary field
            cofactors[r][c] = determinant(matrix_minor(matrix, r, c))
    return transpose(cofactors)


def transpose(matrix: list[list[F]]) -> list[list[F]]:
    return [list(row) for row in zip(*matrix)]


def matrix_is_invertible(matrix: list[list[F]]) -> bool:
    return determinant(matrix) != matrix[0][0].zero()


# random number generator
class Shaker:
    """A class that provides a stream of randomness from a seed"""

    def __init__(self, seed: bytes, buffer_size=2**27) -> None:
        self.data = shake_256(seed).digest(buffer_size)
        self.byte_cnt = 0

    def __call__(self, n: int) -> bytes:
        """returns n bytes of randomness"""
        if len(self.data) < n:
            raise BufferError("Not enough randomness in the pool")

        ret = self.data[:n]
        self.data = self.data[n:]
        self.byte_cnt += n
        return ret


# vision specific functions
def linearized_is_invertible(coefficients: list[F]) -> bool:
    deg: int = coefficients[0].field.degree
    mat: list[list[F]] = [[type(coefficients[0]).zero() for _ in range(deg)] for _ in range(deg)]

    for i in range(min(deg, len(coefficients))):
        mat[0][i] = coefficients[i]

    for i in range(1, deg):
        for j in range(1, deg):
            mat[i][j] = mat[i - 1][j - 1]
        mat[i][0] = mat[i - 1][deg - 1]

    for i in range(1, deg):
        for j in range(deg):
            mat[i][j] = mat[i][j] ** (2**i)

    return matrix_is_invertible(mat)


def affine_inverse(b: list[F]) -> list[F]:
    """Return the inverse of the affine linearized polynomial b"""
    deg = b[0].field.degree

    mat = [[type(b[0]).zero() for _ in range(deg)] for _ in range(deg)]

    mat[0][0] = b[1]
    mat[0][1] = b[2]
    mat[0][2] = b[3]

    for i in range(1, deg):
        for j in range(1, deg):
            mat[i][j] = mat[i - 1][j - 1]
        mat[i][0] = mat[i - 1][deg - 1]

    for i in range(1, deg):
        for j in range(deg):
            mat[i][j] = mat[i][j] ** (2**i)

    mat = matrix_inverse(mat)

    b_i = mat[0]
    b_1 = sum([b_i[i] * b[0] ** (2**i) for i in range(deg)], start=type(b[0]).zero())

    return [b_1] + b_i


def matrix_vector_multiply(matrix: list[list[F]], vector: list[F]) -> list[F]:
    n = len(matrix)
    zero = matrix[0][0].zero()

    result = [zero for _ in range(n)]

    for i in range(n):
        for j in range(n):
            result[i] += matrix[i][j] * vector[j]

    return result


@dataclass(frozen=True)
class VisionConstants(Generic[F]):
    b: list[F]
    b_inv: list[F]
    init_const: list[F]
    matrix_const: list[list[F]]
    const_const: list[F]


def is_mds(matrix: list[list[F]]) -> bool:
    """this will "never" finish for n = 24
    O(sum_{k=1}^n {(n choose k)^2 * k!)}"""
    n = len(matrix)
    elem = type(matrix[0][0])
    for size in range(1, n + 1):
        for rows in combinations(range(n), size):
            for cols in combinations(range(n), size):
                submatrix = [[matrix[r][c] for c in cols] for r in rows]
                if determinant(submatrix) == elem.zero():
                    return False
    return True


def echelon_form(matrix: list[list[F]]) -> list[list[F]]:
    row_num = len(matrix)
    col_num = len(matrix[0])

    head = 0
    zero = matrix[0][0].zero()

    # Row-Echelon form Matrix
    rem = matrix.copy()

    for r in range(row_num):
        if head >= col_num:
            return rem
        i = r
        while rem[i][head] == zero:
            i += 1
            if i == row_num:
                i = r
                head += 1
                if col_num == head:
                    return rem

        rem[i], rem[r] = rem[r], rem[i]
        lv = rem[r][head]
        rem[r] = [mrx * lv.inverse() for mrx in rem[r]]
        # rem[r] = [ mrx / float(lv) for mrx in rem[r]]

        for i in range(row_num):
            if i != r:
                lv = rem[i][head]
                rem[i] = [iv - lv * rv for rv, iv in zip(rem[r], rem[i])]
        head += 1

    return rem


class Bit(BinaryTowerFieldElem):
    field = FanPaarTowerField(0)


def elem2bits(x: F) -> list[Bit]:
    return [Bit((x.value >> j) & 1) for j in range(x.field.degree)]


def bits2elem(x: list[F]) -> Any:
    class Elem(BinaryTowerFieldElem):
        field = FanPaarTowerField(ceil(log2(len(x))))

    return Elem(sum([x[j].value << j for j in range(len(x))]))


def evaluate_affine_linearized_poly(poly: list[F], x: F) -> F:
    result = poly[-1]
    base = x
    for i in range(len(poly) - 1):
        result += poly[i] * base
        base *= base
    return result


def generate_affine_matrix(p: list[F]) -> tuple[list[list[Bit]], list[Bit]]:
    """returns a matrix m and a vector v of the affine linearized polynomial p"""

    Elem = type(p[0])

    offset_vector = elem2bits(p[-1])

    bit_matrix_columns = [
        [a + b for a, b in zip(elem2bits(evaluate_affine_linearized_poly(p, Elem(1 << i))), offset_vector)]
        for i in range(Elem.field.degree)
    ]
    bit_matrix = transpose(bit_matrix_columns)

    return bit_matrix, offset_vector
