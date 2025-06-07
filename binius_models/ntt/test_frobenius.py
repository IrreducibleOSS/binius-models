from ..finite_fields.tower import BinaryTowerFieldElem, FanPaarTowerField
from .additive_ntt import GaoMateerBasis
from .frobenius import FrobeniusNTT


class Elem1bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(0)


class Elem32bFP(BinaryTowerFieldElem):
    field = FanPaarTowerField(5)


def test_frobenius() -> None:
    log_h = 5
    frob = FrobeniusNTT(log_h, 2)
    mateer = GaoMateerBasis(Elem32bFP, log_h, 2)
    input = [Elem1bFP.random() for _ in range(1 << log_h)]
    upcasted = [element.upcast(Elem32bFP) for element in input]
    assert frob.unpack_output(frob.encode(input)) == mateer.encode(upcasted)
