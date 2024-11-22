# (C) 2024 Irreducible Inc.

load("generate_vision_instance.sage")

mark32b = Vision(128, 32, 24, 8)

def dump_1d(name, array):
    print(f"{name}: {', '.join(mark32b.field.to_hex(x) for x in array)}")

def dump_2d(name, matrix):
    print(f"{name}:")
    for row in matrix:
        print(f"[{', '.join(mark32b.field.to_hex(x) for x in row)}]")

def dump_2d_mds(name, matrix):
    print(f"{name}:")
    for row in matrix:
        print(f"[{', '.join(mark32b.mds_field.to_hex(x) for x in row)}]")

dump_1d("B", mark32b.b)
dump_1d("B_inv", mark32b.b_inv)
dump_1d("initial_constant", mark32b.initial_constant)
dump_2d("constants_matrix", mark32b.constants_matrix)
dump_1d("constants_constant", mark32b.constants_constant)
dump_2d_mds("mds", mark32b.mds)

zero = vector([mark32b.field.from_integer(0) for _ in range(mark32b.m)])

mark32b.update_key_schedule(copy(zero))
dump_2d("key_schedule", mark32b.key_schedule)

ciphertext = mark32b.encrypt(copy(zero))
dump_1d("plaintext", zero)
dump_1d("ciphertext", ciphertext)

ciphertext = mark32b.encrypt(mark32b.random_state)
dump_1d("plaintext", mark32b.random_state)
dump_1d("ciphertext", ciphertext)

message = [mark32b.field.from_bytes(b"\xde\xad\xbe\xef", "little")]
hash = mark32b.sponge(message)
dump_1d("message", message)
dump_1d("spongehash", hash)

msg = "You can prove anything you want by coldly logical reason--if you pick the proper postulates."
data = msg.encode("utf-8")
message = [mark32b.field.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]
hash = mark32b.sponge(message)
dump_1d("message", message)
dump_1d("spongehash", hash)
