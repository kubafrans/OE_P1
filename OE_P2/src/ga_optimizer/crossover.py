from __future__ import annotations

import random

from .config import CrossoverMethod


def one_point(parent_a: str, parent_b: str, rng: random.Random) -> tuple[str, str]:
    if len(parent_a) <= 1:
        return parent_a, parent_b
    point = rng.randint(1, len(parent_a) - 1)
    return parent_a[:point] + parent_b[point:], parent_b[:point] + parent_a[point:]


def two_point(parent_a: str, parent_b: str, rng: random.Random) -> tuple[str, str]:
    if len(parent_a) <= 2:
        return one_point(parent_a, parent_b, rng)
    left = rng.randint(1, len(parent_a) - 2)
    right = rng.randint(left + 1, len(parent_a) - 1)
    child_a = parent_a[:left] + parent_b[left:right] + parent_a[right:]
    child_b = parent_b[:left] + parent_a[left:right] + parent_b[right:]
    return child_a, child_b


def uniform(parent_a: str, parent_b: str, rng: random.Random) -> tuple[str, str]:
    child_a = []
    child_b = []
    for bit_a, bit_b in zip(parent_a, parent_b, strict=True):
        if rng.random() < 0.5:
            child_a.append(bit_a)
            child_b.append(bit_b)
        else:
            child_a.append(bit_b)
            child_b.append(bit_a)
    return "".join(child_a), "".join(child_b)


def granular(parent_a: str, parent_b: str, rng: random.Random) -> tuple[str, str]:
    block_size = max(1, len(parent_a) // 4)
    child_a = []
    child_b = []
    use_a = True
    for index in range(0, len(parent_a), block_size):
        slice_a = parent_a[index:index + block_size]
        slice_b = parent_b[index:index + block_size]
        if rng.random() < 0.5:
            use_a = not use_a
        if use_a:
            child_a.append(slice_a)
            child_b.append(slice_b)
        else:
            child_a.append(slice_b)
            child_b.append(slice_a)
    return "".join(child_a), "".join(child_b)


def crossover(parent_a: str, parent_b: str, method: CrossoverMethod, rng: random.Random) -> tuple[str, str]:
    if method == CrossoverMethod.ONE_POINT:
        return one_point(parent_a, parent_b, rng)
    if method == CrossoverMethod.TWO_POINT:
        return two_point(parent_a, parent_b, rng)
    if method == CrossoverMethod.UNIFORM:
        return uniform(parent_a, parent_b, rng)
    return granular(parent_a, parent_b, rng)
