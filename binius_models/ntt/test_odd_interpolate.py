import pytest

from ..finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField, FASTowerField
from .additive_ntt import AdditiveNTT
from .odd_interpolate import InterpolateNonTwoPrimary


class Elem16bFAST(BinaryTowerFieldElem):
    field = FASTowerField(4)


class Elem16bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(4)


# this test will create a random coefficient vector of size d<<ell, then Reed-Solomon encode
# and then take the first d<<ell outputs and interpolate. we check if the result is the same
# as our initial vector.
@pytest.mark.parametrize("Elem16b", [Elem16bFP, Elem16bFAST])
@pytest.mark.parametrize("ell", [3, 5, 7])
@pytest.mark.parametrize("d", [2, 5])
def test_odd_interpolate(Elem16b: type[BinaryTowerFieldElem], ell: int, d: int) -> None:
    interpolate = InterpolateNonTwoPrimary(Elem16b, ell, d)
    log_h = ell + (d - 1).bit_length()
    ntt = AdditiveNTT(Elem16b, log_h, 0)
    input = [interpolate.field.random() if i < d << ell else interpolate.field.zero() for i in range(1 << log_h)]
    encoding = ntt.encode(input)
    assert interpolate.interpolate(encoding[: d << ell]) == input[: d << ell]
