# (C) 2024 Irreducible Inc.

import math
from typing import Literal, TypeVar

T = TypeVar("T")


def bit_range(x: int, up: int, down: int) -> int:
    assert up >= 0 and down >= 0
    assert up >= down
    return (x >> down) & ((1 << (up - down + 1)) - 1)


def bit_reverse(x: int, length: int) -> int:
    xr = 0
    for i in range(length):
        xr = xr | (bit_range(x, length - 1 - i, length - 1 - i) << i)
    return xr


def bit_reverse_indices(a: list[T]) -> list[T]:
    bit_length = int(math.log2(len(a)))
    assert len(a) == 2**bit_length
    return [a[bit_reverse(i, bit_length)] for i in range(len(a))]


def roll(seq: list, n: int = 1, left: bool = False) -> list:
    """Roll (rotate) a sequence. Rolls to the right by default."""
    n = n % len(seq)
    if left is False:
        n = -n
    return seq[n:] + seq[:n]


def seq2bytes(seq: list[int], bytewidth: int = 8, byteorder: Literal["little", "big"] = "little") -> bytes:
    return b"".join([data.to_bytes(bytewidth, byteorder=byteorder) for data in seq])


def bytes2seq(ba: bytearray, bytewidth: int = 8, byteorder: Literal["little", "big"] = "little") -> list[int]:
    return [int.from_bytes(ba[i : i + bytewidth], byteorder=byteorder) for i in range(0, len(ba), bytewidth)]


def transpose_bits(x: int, n_lo_bits: int, n_hi_bits: int) -> int:
    lo_bits = x & ((1 << n_lo_bits) - 1)
    hi_bits = (x >> n_lo_bits) & ((1 << n_hi_bits) - 1)
    return (lo_bits << n_hi_bits) | hi_bits


def bits_mask(n_bits: int) -> int:
    """Returns a mask for the least-significant bits.

    For example, bits_mask(4) returns 0x0f and bits_mask(9) returns 0x01ff.

    :param n_bits: the number of bits which will be 1.
    """
    return (1 << n_bits) - 1


def is_bit_set(x: int, i: int) -> bool:
    return (x >> i) & 1 != 0


def int_to_bits(x: int, n_bits: int) -> list[int]:
    return [(x >> i) & 1 for i in range(n_bits)]


def trial_divide(n):  # naivest possible factoring alg...
    # returns the smallest divisor of n which is > 1.
    for i in range(2, math.floor(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


def factorize(n):
    factors = []
    while n > 1:
        factor = trial_divide(n)
        factors.append(factor)
        n //= factor
    return factors


def is_power_of_two(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0
