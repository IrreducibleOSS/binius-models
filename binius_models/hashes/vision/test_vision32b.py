# (C) 2024 Irreducible Inc.

import pytest

from binius_models.hashes.vision.vision import Vision32b


# fmt:off
@pytest.mark.parametrize("plaintext, ciphertext", [
    ([0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000],
     [0xc12d2bd5, 0x7b69d9d2, 0x5d706d99, 0x24ac5453, 0xfd9a0f9f, 0xfb7a3be4, 0xd935a59f, 0x691667d7,
      0xb89ee8f9, 0x6fca1714, 0x530caee0, 0x3f884b18, 0x6d6c9a40, 0x940b294c, 0xc9976c5d, 0xb3029e46,
      0x8f5dd6ab, 0x8adc785e, 0x15f0cba2, 0x32cd00a1, 0x5c4ad8b5, 0xd9ce2f5b, 0x8e9d1b5b, 0xe70a9e14]),
    ([0xc45cab27, 0x9b735f2e, 0x090d2a66, 0x140b8ff3, 0xe7778ea8, 0xc19ae0a1, 0xc2ffa056, 0x03b13619,
      0xc7a95da5, 0xe456a375, 0xd1d0a869, 0xe80ce171, 0x12c89a69, 0xada9072e, 0xc7a50ccf, 0xb9ce31bd,
      0xf7e16524, 0xb77c94b4, 0xd98d20aa, 0x8bad80a7, 0xe5f60fed, 0x9e082cfb, 0x1747575c, 0xbcaccaeb],
     [0xbbb328cc, 0x71d93b65, 0x473a5ef4, 0x5185446e, 0xbd3f85c5, 0x1fabd2fb, 0xa62cd343, 0xfe07c31f,
      0x4475c510, 0xee6b377d, 0x7beb8619, 0x28a605bf, 0x3d184345, 0x86b5f751, 0x1c02f436, 0x40562d71,
      0xef782b79, 0x77bb5244, 0x1458083c, 0x7306f3ae, 0x12a17146, 0x8202e4c3, 0x0381f579, 0x27628dc6]),
], ids=["zero", "random"])
# fmt:on
def test_vision32b(plaintext: list[int], ciphertext: list[int]):
    ctx = Vision32b()

    ctx.set_zero_schedule()

    plaintext_ = [ctx.elem(x) for x in plaintext]
    ciphertext_ = [ctx.elem(x) for x in ciphertext]

    ciphertext__ = ctx.encrypt(plaintext_)
    assert ciphertext_ == ciphertext__


def test_simple():
    ctx = Vision32b()
    ctx.set_zero_schedule()
    out = ctx.sponge_hash([ctx.elem.from_bytes(b"\xde\xad\xbe\xef")])
    expected = bytes.fromhex("69e1764144099730124ab8ef1414570895ae9de0b74dedf364c72d118851cf65")
    expected_as_elems = [ctx.elem.from_bytes(expected[i : i + 4]) for i in range(0, len(expected), 4)]
    assert expected_as_elems == out


@pytest.mark.slow
def test_multi_aligned():
    ctx = Vision32b()
    ctx.set_zero_schedule()

    msg = "One part of the mysterious existence of Captain Nemo had been unveiled and, if his identity had not been "
    msg += "recognised, at least, the nations united against him were no longer hunting a chimerical creature, but a "
    msg += "man who had vowed a deadly hatred against them"
    data = msg.encode("utf-8")
    message = [ctx.elem.from_bytes(data[i : i + 4]) for i in range(0, len(data), 4)]
    out = ctx.sponge_hash(message)
    expected = bytes.fromhex("6ade8ba2a45a070a3abaff6f1bf9483686c78d4afca2d0d8d3c7897fdfe2df91")
    expected_as_elems = [ctx.elem.from_bytes(expected[i : i + 4]) for i in range(0, len(expected), 4)]
    assert expected_as_elems == out


@pytest.mark.slow
def test_multi_unaligned():
    ctx = Vision32b()
    ctx.set_zero_schedule()

    msg = "You can prove anything you want by coldly logical reason--if you pick the proper postulates."
    data = msg.encode("utf-8")
    message = [ctx.elem.from_bytes(data[i : i + 4]) for i in range(0, len(data), 4)]
    out = ctx.sponge_hash(message)
    expected = bytes.fromhex("2819814fd9da83ab358533900adaf87f4c9e0f88657f572a9a6e83d95b88a9ea")
    expected_as_elems = [ctx.elem.from_bytes(expected[i : i + 4]) for i in range(0, len(expected), 4)]
    assert expected_as_elems == out
