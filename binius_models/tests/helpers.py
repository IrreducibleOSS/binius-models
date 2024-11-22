# (C) 2024 Irreducible Inc.

from hypothesis import strategies as st


def random_integers_strategy(
    min_value: int,
    max_value: int,
) -> st.SearchStrategy[int]:
    return st.builds(lambda rng: rng.randint(min_value, max_value), st.randoms(use_true_random=True))
