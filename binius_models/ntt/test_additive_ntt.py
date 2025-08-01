import pytest

from ..finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField, FASTowerField
from .additive_ntt import (
    AdditiveNTT,
    CantorAdditiveNTT,
    FourStepAdditiveNTT,
    GaoMateerBasis,
)


class Elem8bFAST(BinaryTowerFieldElem):
    field = FASTowerField(3)


class Elem8bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(3)


class Elem16bFAST(BinaryTowerFieldElem):
    field = FASTowerField(4)


class Elem16bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(4)


class Elem32bFAST(BinaryTowerFieldElem):
    field = FASTowerField(5)


class Elem32bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(5)


@pytest.mark.parametrize("Elem8b", [Elem8bFP, Elem8bFAST])
def test_ntt(Elem8b: type[BinaryTowerFieldElem]) -> None:
    # length 2⁵, rate 1/4, so 4× in length. note that the block length is only 2⁷ here;
    # the code will "intelligently" know to only do this over the smaller field 𝔽_{2⁸}.
    log_h = 5
    ntt = AdditiveNTT(Elem8b, log_h, 2)
    input = [ntt.field.random() for _ in range(1 << log_h)]
    assert ntt.encode(input) == ntt._naive_encode(input)


@pytest.mark.slow
@pytest.mark.parametrize("Elem16b", [Elem16bFP, Elem16bFAST])
def test_ntt_large(Elem16b: type[BinaryTowerFieldElem]) -> None:
    log_h = 7
    ntt = AdditiveNTT(Elem16b, log_h, 2)
    input = [ntt.field.random() for _ in range(1 << log_h)]
    assert ntt.encode(input) == ntt._naive_encode(input)


@pytest.mark.parametrize("Elem16b", [Elem16bFP, Elem16bFAST])
def test_four_step_large(Elem16b: type[BinaryTowerFieldElem]) -> None:
    # start with length 2⁷ = 128; 4x it in length, ending with 512
    log_h = 7
    ntt = AdditiveNTT(Elem16b, log_h, 2)
    four_step = FourStepAdditiveNTT(Elem16b, log_h, 2)
    input = [ntt.field.random() for _ in range(1 << log_h)]
    assert four_step.encode(input) == ntt.encode(input)


@pytest.mark.parametrize("Elem16b", [Elem16bFP, Elem16bFAST])
def test_four_step_larger(Elem16b: type[BinaryTowerFieldElem]) -> None:
    # start with length 2⁸ = 256; 16x it in length, ending with 4096
    log_h = 8
    ntt = AdditiveNTT(Elem16b, log_h, 4)
    four_step = FourStepAdditiveNTT(Elem16b, log_h, 4)
    input = [ntt.field.random() for _ in range(1 << log_h)]
    assert four_step.encode(input) == ntt.encode(input)


@pytest.mark.parametrize("Elem8b", [Elem8bFP, Elem8bFAST])
def test_high_to_low(Elem8b: type[BinaryTowerFieldElem]) -> None:
    # length 2⁵, rate 1/4, so 4× in length. note that the block length is only 2⁷ here;
    # the code will "intelligently" know to only do this over the smaller field 𝔽_{2⁸}.
    log_h = 5
    ntt = AdditiveNTT(Elem8b, log_h, 0, high_to_low=True)
    input = [ntt.field.random() for _ in range(1 << log_h)]
    assert ntt._inverse_transform(ntt.encode(input), 0) == input  # just test bijection; real test will be use in Frib


def test_cantor() -> None:
    log_h = 5
    ntt = CantorAdditiveNTT(Elem16bFAST, log_h, 2)
    input = [ntt.field.random() for _ in range(1 << log_h)]
    assert ntt.encode(input) == ntt._naive_encode(input)


@pytest.mark.slow
def test_cantor_fail() -> None:
    log_h = 5
    ntt = CantorAdditiveNTT(Elem16bFP, log_h, 2)
    # this should NOT work, since Cantor NTT needs a FAST field, not an arbitrary (e.g. Fan–Paar) field.
    input = [ntt.field.random() for _ in range(1 << log_h)]
    assert ntt.encode(input) != ntt._naive_encode(input)


def test_inverse_smaller_input() -> None:
    max_log_h = 5
    log_h = 3
    log_inv_rate = 2
    ntt = AdditiveNTT(Elem8bFP, max_log_h, log_inv_rate)
    input = [ntt.field.random() for _ in range(1 << log_h)]
    intermediate = ntt._inverse_transform(input, 0)
    output = ntt._transform(intermediate, 0)
    assert input == output


def test_encode_smaller_input() -> None:
    max_log_h = 5
    log_h = 3
    log_inv_rate = 1
    ntt = AdditiveNTT(Elem32bFP, max_log_h, log_inv_rate)
    input = [ntt.field.random() for _ in range(1 << log_h)]
    assert ntt.encode(input) == ntt._naive_encode(input)


def test_inverse_interleaved() -> None:
    log_h = 5
    log_inv_rate = 2
    tiling_factor = 1
    ntt = AdditiveNTT(Elem8bFP, log_h, log_inv_rate)
    skip_ntt = AdditiveNTT(Elem8bFP, log_h, log_inv_rate, skip_rounds=tiling_factor)
    small = [ntt.field.random() for _ in range(1 << log_h - tiling_factor)]
    tiled: list[Elem8bFP] = sum(([small[i]] * (1 << tiling_factor) for i in range(1 << log_h - tiling_factor)), [])
    small_inverse = skip_ntt._inverse_transform(small, 0)
    tricky_inverse: list[Elem8bFP] = sum(
        (
            [small_inverse[i]] + [Elem8bFP.zero()] * ((1 << tiling_factor) - 1)
            for i in range(1 << log_h - tiling_factor)
        ),
        [],
    )
    naive_inverse = ntt._inverse_transform(tiled, 0)
    assert tricky_inverse == naive_inverse


def test_gao_mateer() -> None:
    max_log_h = 7
    log_h = 5
    log_inv_rate = 2
    mateer = GaoMateerBasis(Elem32bFP, max_log_h, log_inv_rate)  # to get "full" basis, run w/ max_log_h + rate == 32
    input = [Elem32bFP.random() for _ in range(1 << log_h)]
    assert mateer.encode(input) == mateer._naive_encode(input)
