# (C) 2024 Irreducible Inc.

from typing import TypeVar

from pytest import mark

from binius_models.hashes.vision.utils import (
    bits2elem,
    elem2bits,
    evaluate_affine_linearized_poly,
    generate_affine_matrix,
    matrix_vector_multiply,
)
from binius_models.hashes.vision.vision import Vision, Vision32b

H = TypeVar("H", bound="Vision")
TEST_NUM = 2**8


def py2sv(s: str) -> str:
    return ", ".join([x.replace("0x", "32'h") for x in reversed(s.split(", "))])


@mark.parametrize("instance", [Vision32b()], ids=["Vision32b"])
def test_generate_affine_matrix(instance: H) -> None:
    print()

    matrix_b, offset_b = generate_affine_matrix(instance.constants.b)

    for _ in range(TEST_NUM):
        x = instance.elem.random()
        y = evaluate_affine_linearized_poly(instance.constants.b, x)

        x_bits = elem2bits(x)
        xm_bits = matrix_vector_multiply(matrix_b, x_bits)
        y_bits = [a + b for a, b in zip(xm_bits, offset_b)]

        assert y_bits == elem2bits(y)

    # print("matrix_b:")
    # for row in matrix_b:
    #     print(", ".join([str(x) for x in row]))

    # print("offset_b:")
    # print(", ".join([str(x) for x in offset_b]))

    print("matrix_b:")
    print(py2sv(", ".join([str(bits2elem(row)) for row in matrix_b])))
    print("offset_b:")
    print(bits2elem(offset_b))

    matrix_b_inv, offset_b_inv = generate_affine_matrix(instance.constants.b_inv)

    for _ in range(TEST_NUM):
        x = instance.elem.random()
        y = evaluate_affine_linearized_poly(instance.constants.b_inv, x)

        x_bits = elem2bits(x)
        xm_bits = matrix_vector_multiply(matrix_b_inv, x_bits)
        y_bits = [a + b for a, b in zip(xm_bits, offset_b_inv)]

        assert y_bits == elem2bits(y)

    # print("matrix_b_inv:")
    # for row in matrix_b_inv:
    #     print(", ".join([str(x) for x in row]))

    # print("offset_b_inv:")
    # print(", ".join([str(x) for x in offset_b_inv]))

    print("matrix_b_inv:")
    print(py2sv(", ".join([str(bits2elem(row)) for row in matrix_b_inv])))
    print("offset_b_inv:")
    print(bits2elem(offset_b_inv))
