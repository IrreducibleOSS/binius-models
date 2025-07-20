from typing import TypeVar

from ..finite_fields.finite_field import FiniteFieldElem

F = TypeVar("F", bound=FiniteFieldElem)


def tensor_expand(field: type[F], x: list[F], vars: int) -> list[F]:
    result = [field.one()] * (1 << vars)
    for i in range(vars):
        for j in range(1 << i):
            result[1 << i | j] = result[j] * x[i]
            result[j] -= result[1 << i | j]
    return result


def univariate_expand(field, r_zhat_prime):
    # computes the list L̃(r'_ẑ, i), for i ∈ {0, … , 63}. the standard barycentric approach is to realize that,
    # for each i, L̃(r'_ẑ, i) == ∏ⱼ(r'_ẑ – j) ⋅ [ w[i] / (r'_ẑ – i) ], where w[i] are the barycentric weights.
    # the left-hand product ∏ⱼ( r'_ẑ – j) doesn't depend on i, so we can compute it once, using n – 1 mults,
    # and then finish the job using n mults and n divisions. total n - 1 + 2 ⋅ n == 3 ⋅ n - 1 mults and n inversions

    # we are going to be slightly cleverer. for each i, we want the all-but-one product p[i] ≔ ∏_{j ≠ i} (r'_ẑ – j)
    # if we have those, we can finish the job w/ n further mults, by setting L̃(r'_ẑ, i) ≔ p[i] ⋅ w[i] for each i.
    # we can get the all-but-one products with 2 ⋅ n space overhead by precomputing the prefix and suffix products
    # when all is said and done, we get 2 ⋅ (n – 1) + 2 ⋅ n = 4 ⋅ n - 2 mults, so we replaced invs --> mults
    # the problem size is small---but see later whether we end up needing this in any "hot" parts of the code?

    # begin computation of barycentric weight (!). you read that right---just one weight, not many.
    # it's a fact that when the interpolation domain is a power of 2, then all barycentric weights wind up identical.
    # actually, the operative condition is that the interpolation domain is an additive subgroup of the field.
    # if the domain is of size n == 2ᵏ, then for each i ∈ {0, … , 2ᵏ − 1}, { i ⊕ j | j ≠ i } == {1, … , 2ᵏ − 1} (!).
    # indeed for each i, xor-with-i is a bijection on {0, … , 2ᵏ − 1} ≅ ℤ₂ᵏ. its image, \ {0}, is {1, … , 2ᵏ − 1}.
    # it's enough to take the product of the elements of {1, … , 2ᵏ − 1}, and then invert that, to get our "weight".
    w = field(1)
    for i in range(1, 64):
        w *= field(i)
    w = w.inverse()

    prefixes = [field.one()] * 64
    suffixes = [field.one()] * 64
    for i in range(1, 64):
        prefixes[i] = prefixes[i - 1] * (r_zhat_prime - field(i - 1))
        suffixes[64 - 1 - i] = suffixes[64 - i] * (r_zhat_prime - field(64 - i))
    result = [field.zero()] * 64
    for i in range(64):
        result[i] = prefixes[i] * suffixes[i] * w  # these are all the same....! can you save?
    return result
