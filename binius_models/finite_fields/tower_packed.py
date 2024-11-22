# (C) 2024 Irreducible Inc.

from __future__ import annotations


def multiply_simd(height: int, log_width: int, a: int, b: int) -> int:
    """
    Interpret the inputs a and b as packed vectors of field elements and multiply them in parallel.

    The parameters a and b are bit-strings viewed as a vector of field elements in little-endian
    order. This multiplies them in SIMD fashion, returning the packed product.

    :param height: the tower height
    :param log_width: base-2 logarithm of the number of packed elements
    :param a: packed bit representation of first multiplicand vector in little-endian order
    :param b: packed bit representation of second multiplicand vector in little-endian order
    :returns: packed bit representation of product vector in little-endian order
    """

    # In T_0, the product is just bitwise and
    if height == 0:
        return a & b

    # a and b can be interpreted as packed subfield elements:
    # a = <a_lo_0, a_hi_0, a_lo_1, a_hi_1, ...>
    # b = <b_lo_0, b_hi_0, b_lo_1, b_hi_1, ...>

    # ab is the product of a * b as packed subfield elements
    # ab = <a_lo_0 * b_lo_0, a_hi_0 * b_hi_0, a_lo_1 * b_lo_1, a_hi_1 * b_hi_1, ...>
    z0_even_z2_odd = multiply_simd(height - 1, log_width + 1, a, b)

    # lo = <a_lo_0, b_lo_0, a_lo_1, b_lo_1, ...>
    # hi = <a_hi_0, b_hi_0, a_hi_1, b_hi_1, ...>
    lo, hi = interleave(height - 1, log_width + 1, a, b)

    # <a_lo_0 + a_hi_0, b_lo_0 + b_hi_0, a_lo_1 + a_hi_1, b_lo_1 + b_hi_1, ...>
    lo_plus_hi_a_even_b_odd = lo ^ hi

    block_len = 1 << (height - 1)
    even_mask = _generate_interleave_mask(height - 1, log_width + 1)
    odd_mask = even_mask << block_len

    alphas = _generate_alphas_even(height - 1, log_width + 1)

    # <α, z2_0, α, z2_1, ...>
    alpha_even_z2_odd = alphas ^ (z0_even_z2_odd & odd_mask)

    # a_lo_plus_hi_even_z2_odd    = <a_lo_0 + a_hi_0, z2_0, a_lo_1 + a_hi_1, z2_1, ...>
    # b_lo_plus_hi_even_alpha_odd = <b_lo_0 + b_hi_0,    α, a_lo_1 + a_hi_1,   αz, ...>
    a_lo_plus_hi_even_alpha_odd, b_lo_plus_hi_even_z2_odd = interleave(
        height - 1, log_width + 1, lo_plus_hi_a_even_b_odd, alpha_even_z2_odd
    )
    # <z1_0 + z0_0 + z2_0, z2a_0, z1_1 + z0_1 + z2_1, z2a_1, ...>
    z1_plus_z0_plus_z2_even_z2a_odd = multiply_simd(
        height - 1, log_width + 1, a_lo_plus_hi_even_alpha_odd, b_lo_plus_hi_even_z2_odd
    )

    # <0, z1_0 + z2a_0 + z0_0 + z2_0, 0, z1_1 + z2a_1 + z0_1 + z2_1, ...>
    zero_even_z1_plus_z2a_plus_z0_plus_z2_odd = (
        z1_plus_z0_plus_z2_even_z2a_odd ^ (z1_plus_z0_plus_z2_even_z2a_odd << block_len)
    ) & odd_mask

    # <z0_0 + z2_0, z0_0 + z2_0, z0_1 + z2_1, z0_1 + z2_1, ...>
    z0_plus_z2_dup = xor_adjacent(height - 1, log_width + 1, z0_even_z2_odd)

    # <z0_0 + z2_0, z1_0 + z2a_0, z0_1 + z2_1, z1_1 + z2a_1, ...>
    return z0_plus_z2_dup ^ zero_even_z1_plus_z2a_plus_z0_plus_z2_odd


def interleave(height: int, log_width: int, a: int, b: int) -> tuple[int, int]:
    """View the inputs as vectors of packed binary tower elements and transpose as 2x2 square matrices.

    Given vectors <a_0, a_1, a_2, a_3, ...> and <b_0, b_1, b_2, b_3, ...>, returns a tuple with
    <a0, b0, a2, b2, ...> and <a1, b1, a3, b3>.
    """

    if log_width <= 0:
        raise ValueError("log_width must be greater than 0")

    # See Hacker's Delight, Section 7-3.
    # https://dl.acm.org/doi/10.5555/2462741

    # The interleave mask can be precomputed and stored in a static array when the total width
    # (2**(height + log_width)) is a constant.
    mask = _generate_interleave_mask(height, log_width)

    block_len = 1 << height
    t = ((a >> block_len) ^ b) & mask
    c = a ^ (t << block_len)
    d = b ^ t
    return c, d


def xor_adjacent(height: int, log_width: int, a: int) -> int:
    """View the input as a vector of packed binary tower elements and add the adjacent ones.

    Given a vector <a_0, a_1, a_2, a_3, ...>, returns <a0 + a1, a0 + a1, a2 + a3, a2 + a3, ...>.
    """

    if log_width <= 0:
        raise ValueError("log_width must be greater than 0")

    # See Hacker's Delight, Section 7-3.
    # https://dl.acm.org/doi/10.5555/2462741

    # The interleave mask can be precomputed and stored in a static array when the total width
    # (2**(height + log_width)) is a constant.
    mask = _generate_interleave_mask(height, log_width)

    block_len = 1 << height
    t = ((a >> block_len) ^ a) & mask
    return t ^ (t << block_len)


def _flip(height: int, log_width: int, a: int) -> int:
    """View the input as a vector of packed binary tower elements and flip the adjacent ones.

    Given a vector <a_0, a_1, a_2, a_3, ...>, returns <a1, a0, a3, a2, ...>.
    """

    if log_width <= 0:
        raise ValueError("log_width must be greater than 0")

    # The interleave mask can be precomputed and stored in a static array when the total width
    # (2**(height + log_width)) is a constant.
    mask = _generate_interleave_mask(height, log_width)

    block_len = 1 << height
    t = ((a >> block_len) ^ a) & mask
    return a ^ ((t << block_len) ^ t)


def _generate_interleave_mask(height: int, log_width: int) -> int:
    block_len = 1 << height
    mask = (1 << block_len) - 1
    for i in range(1, log_width):
        mask |= mask << (1 << (height + i))
    return mask


def _generate_alphas_even(height: int, log_width: int) -> int:
    """Generate the packed value with alpha in the even positions and zero in the odd positions."""
    if log_width <= 0:
        raise ValueError("log_width must be greater than 0")

    if height == 0:
        alphas = 1
    else:
        alphas = 1 << (1 << (height - 1))

    for i in range(1, log_width):
        alphas |= alphas << (1 << (height + i))
    return alphas


def _multiply_alpha_simd(height: int, log_width: int, a: int) -> int:
    """
    Interpret the input a as a packed vector of field elements and multiply by alpha in parallel.

    The parameter a is a bit-string viewed as a vector of field elements in little-endian
    order. This multiplies it by alpha in SIMD fashion, returning the packed product.

    :param height: the tower height
    :param log_width: base-2 logarithm of the number of packed elements
    :param a: packed bit representation of a vector in little-endian order
    :returns: packed bit representation of product by alpha vector in little-endian order
    """

    if height == 0:
        return a

    block_len = 1 << (height - 1)
    even_mask = _generate_interleave_mask(height - 1, log_width + 1)
    odd_mask = even_mask << block_len

    a0 = a & even_mask
    a1 = a & odd_mask
    z1 = _multiply_alpha_simd(height - 1, log_width + 1, a1)

    return (a1 >> block_len) | ((a0 << block_len) ^ z1)


def square_simd(height: int, log_width: int, a: int) -> int:
    """
    Interpret the input a as a packed vector of field elements and multiply it by itself in parallel.

    The parameter a is a bit-string viewed as a vector of field elements in little-endian
    order. This multiplies it by itself in SIMD fashion, returning the packed result.

    :param height: the tower height
    :param log_width: base-2 logarithm of the number of packed elements
    :param a: packed bit representation of a vector in little-endian order
    :returns: packed bit representation of vector of square values in little-endian order
    """

    if height == 0:
        return a

    block_len = 1 << (height - 1)
    even_mask = _generate_interleave_mask(height - 1, log_width + 1)
    odd_mask = even_mask << block_len

    z_02 = square_simd(height - 1, log_width + 1, a)
    z_2a = _multiply_alpha_simd(height - 1, log_width + 1, z_02) & odd_mask

    z_0_xor_z_2 = (z_02 ^ (z_02 >> block_len)) & even_mask

    return z_0_xor_z_2 | z_2a


def inverse_simd(height: int, log_width: int, a: int) -> int:
    """
    Interpret the input a as a packed vector of field elements and calculate inverse values in parallel.

    The parameter a is a bit-string viewed as a vector of field elements in little-endian
    order. This calculates the inverse value in SIMD fashion, returning the vector of inverse values.
    The places where the original values were zero will be filled with zeroes.

    :param height: the tower height
    :param log_width: base-2 logarithm of the number of packed elements
    :param a: packed bit representation of a vector in little-endian order
    :returns: packed bit representation of vector of inverse values in little-endian order.
    """
    if height == 0:
        return a

    block_len = 1 << (height - 1)
    even_mask = _generate_interleave_mask(height - 1, log_width + 1)
    odd_mask = even_mask << block_len

    # has right values in even positions
    a_1_even = a >> block_len
    intermediate = a ^ _multiply_alpha_simd(height - 1, log_width + 1, a_1_even)
    delta = multiply_simd(height - 1, log_width + 1, a, intermediate) ^ square_simd(height - 1, log_width + 1, a_1_even)
    delta_inv = inverse_simd(height - 1, log_width + 1, delta)

    # set values from even positions to odd as well
    delta_inv_delta_inv = delta_inv & even_mask
    delta_inv_delta_inv |= delta_inv_delta_inv << block_len

    return multiply_simd(height - 1, log_width + 1, delta_inv_delta_inv, (a & odd_mask) | (intermediate & even_mask))


# FASTowerField


def fast_tower_field_multiply_simd(height: int, log_width: int, a: int, b: int) -> int:
    """
    Interpret the inputs a and b as packed vectors of field elements and multiply them in parallel.

    The parameters a and b are bit-strings viewed as a vector of field elements in little-endian
    order. This multiplies them in SIMD fashion, returning the packed product.

    :param height: the tower height
    :param log_width: base-2 logarithm of the number of packed elements
    :param a: packed bit representation of first multiplicand vector in little-endian order
    :param b: packed bit representation of second multiplicand vector in little-endian order
    :returns: packed bit representation of product vector in little-endian order
    """

    # In T_0, the product is just bitwise and
    if height == 0:
        return a & b

    # a and b can be interpreted as packed subfield elements:
    # a = <a_lo_0, a_hi_0, a_lo_1, a_hi_1, ...>
    # b = <b_lo_0, b_hi_0, b_lo_1, b_hi_1, ...>

    # ab is the product of a * b as packed subfield elements
    # ab = <a_lo_0 * b_lo_0, a_hi_0 * b_hi_0, a_lo_1 * b_lo_1, a_hi_1 * b_hi_1, ...>
    z0_even_z2_odd = fast_tower_field_multiply_simd(height - 1, log_width + 1, a, b)

    # lo = <a_lo_0, b_lo_0, a_lo_1, b_lo_1, ...>
    # hi = <a_hi_0, b_hi_0, a_hi_1, b_hi_1, ...>
    lo, hi = interleave(height - 1, log_width + 1, a, b)

    # <a_lo_0 + a_hi_0, b_lo_0 + b_hi_0, a_lo_1 + a_hi_1, b_lo_1 + b_hi_1, ...>
    lo_plus_hi_a_even_b_odd = lo ^ hi

    block_len = 1 << (height - 1)
    even_mask = _generate_interleave_mask(height - 1, log_width + 1)
    odd_mask = even_mask << block_len

    alphas = _fast_tower_field_generate_alphas_even(height - 1, log_width + 1)

    # <α, z2_0, α, z2_1, ...>
    alpha_even_z2_odd = alphas ^ (z0_even_z2_odd & odd_mask)

    # alpha_even_a_lo_plus_hi_odd = <   α, a_lo_0 + a_hi_0,    α, ...>
    # z2_even_b_lo_plus_hi_odd    = <z2_0, b_lo_0 + b_hi_0, z2_1, ...>
    alpha_even_a_lo_plus_hi_odd, z2_even_b_lo_plus_hi_odd = interleave(
        height - 1, log_width + 1, alpha_even_z2_odd, lo_plus_hi_a_even_b_odd
    )

    # <z2a_0, z1_0 + z0_0, z2a_1, ...>
    z2a_even_z1_plus_z0_odd = fast_tower_field_multiply_simd(
        height - 1, log_width + 1, alpha_even_a_lo_plus_hi_odd, z2_even_b_lo_plus_hi_odd
    )

    # <z0_0, 0, z0_1, 0, ...>
    z0_even_zero_odd = z0_even_z2_odd & even_mask

    # <z0_0, z0_0, z0_1, z0_1, ...>
    z0_dup1 = (z0_even_zero_odd << block_len) ^ z0_even_zero_odd

    # <z0_0 + z2a_0, z1_0, z0_1 + z2a_1, z1_1, ...>
    return z0_dup1 ^ z2a_even_z1_plus_z0_odd


def _fast_tower_field_generate_alphas_even(height: int, log_width: int) -> int:
    """Generate the packed value with alpha in the even positions and zero in the odd positions."""
    if log_width <= 0:
        raise ValueError("log_width must be greater than 0")

    if height == 0:
        alphas = 1
    else:
        alphas = 1 << ((1 << height) - 1)
    for i in range(1, log_width):
        alphas |= alphas << (1 << (height + i))
    return alphas


def _fast_tower_field_multiply_alpha_simd(height: int, log_width: int, a: int) -> int:
    """
    Interpret the input a as a packed vector of field elements and multiply by alpha in parallel.

    The parameter a is a bit-string viewed as a vector of field elements in little-endian
    order. This multiplies it by alpha in SIMD fashion, returning the packed product.

    :param height: the tower height
    :param log_width: base-2 logarithm of the number of packed elements
    :param a: packed bit representation of a vector in little-endian order
    :returns: packed bit representation of product by alpha vector in little-endian order
    """

    if height == 0:
        return a

    block_len = 1 << (height - 1)
    even_mask = _generate_interleave_mask(height - 1, log_width + 1)
    odd_mask = even_mask << block_len

    a0 = a & even_mask
    a1 = a & odd_mask
    z2 = _fast_tower_field_multiply_alpha_simd(height - 1, log_width + 1, a1)
    z1 = _fast_tower_field_multiply_alpha_simd(height - 1, log_width + 1, a0)
    z2a = _fast_tower_field_multiply_alpha_simd(height - 1, log_width + 1, z2)

    return (z2a >> block_len) | ((z1 << block_len) ^ z2)


def fast_tower_field_square_simd(height: int, log_width: int, a: int) -> int:
    """
    Interpret the input a as a packed vector of field elements and multiply it by itself in parallel.

    The parameter a is a bit-string viewed as a vector of field elements in little-endian
    order. This multiplies it by itself in SIMD fashion, returning the packed result.

    :param height: the tower height
    :param log_width: base-2 logarithm of the number of packed elements
    :param a: packed bit representation of a vector in little-endian order
    :returns: packed bit representation of vector of square values in little-endian order
    """

    if height == 0:
        return a

    block_len = 1 << (height - 1)
    even_mask = _generate_interleave_mask(height - 1, log_width + 1)

    z0_even_z2_odd = fast_tower_field_square_simd(height - 1, log_width + 1, a)
    z2a_even_zero_odd = (
        _fast_tower_field_multiply_alpha_simd(height - 1, log_width + 1, z0_even_z2_odd) >> block_len
    ) & even_mask

    return z0_even_z2_odd ^ z2a_even_zero_odd


def fast_tower_field_inverse_simd(height: int, log_width: int, a: int) -> int:
    """
    Interpret the input a as a packed vector of field elements and calculate inverse values in parallel.

    The parameter a is a bit-string viewed as a vector of field elements in little-endian
    order. This calculates the inverse value in SIMD fashion, returning the vector of inverse values.
    The places where the original values were zero will be filled with zeroes.

    :param height: the tower height
    :param log_width: base-2 logarithm of the number of packed elements
    :param a: packed bit representation of a vector in little-endian order
    :returns: packed bit representation of vector of inverse values in little-endian order.
    """
    if height == 0:
        return a

    block_len = 1 << (height - 1)
    even_mask = _generate_interleave_mask(height - 1, log_width + 1)
    odd_mask = even_mask << block_len

    a_0_xor_a_1_even_zero_odd = (a ^ a >> block_len) & even_mask
    # has right values in even positions
    a_1_even = a >> block_len

    a_1_even_square = fast_tower_field_square_simd(height - 1, log_width + 1, a_1_even)
    a_1_even_square_alpha = _fast_tower_field_multiply_alpha_simd(height - 1, log_width + 1, a_1_even_square)

    delta = (
        fast_tower_field_multiply_simd(height - 1, log_width + 1, a, a_0_xor_a_1_even_zero_odd) ^ a_1_even_square_alpha
    )
    delta_inv = fast_tower_field_inverse_simd(height - 1, log_width + 1, delta)

    # set values from even positions to odd as well
    delta_inv_delta_inv = delta_inv & even_mask
    delta_inv_delta_inv |= delta_inv_delta_inv << block_len

    return fast_tower_field_multiply_simd(
        height - 1, log_width + 1, delta_inv_delta_inv, (a & odd_mask) | (a_0_xor_a_1_even_zero_odd)
    )
