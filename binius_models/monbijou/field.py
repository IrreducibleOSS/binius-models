import galois

field = galois.GF(2**128, irreducible_poly="x^128 + x^7 + x^2 + x + 1")  # GHASH
