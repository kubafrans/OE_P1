from __future__ import annotations

import random

from .config import MutationMethod
from .encoding import BinaryEncoding


def flip_bit(bit: str) -> str:
    return "1" if bit == "0" else "0"


def one_point(chromosome: str, rng: random.Random) -> str:
    if not chromosome:
        return chromosome
    index = rng.randrange(len(chromosome))
    return chromosome[:index] + flip_bit(chromosome[index]) + chromosome[index + 1 :]


def two_point(chromosome: str, rng: random.Random) -> str:
    if len(chromosome) < 2:
        return one_point(chromosome, rng)
    first = rng.randrange(len(chromosome))
    second = rng.randrange(len(chromosome))
    while second == first:
        second = rng.randrange(len(chromosome))
    bits = list(chromosome)
    bits[first] = flip_bit(bits[first])
    bits[second] = flip_bit(bits[second])
    return "".join(bits)


def boundary(chromosome: str, encoding: BinaryEncoding, rng: random.Random) -> str:
    if len(chromosome) < encoding.bits_per_gene:
        return chromosome
    gene_index = rng.randrange(encoding.range_config.dimensions)
    start = gene_index * encoding.bits_per_gene
    end = start + encoding.bits_per_gene
    replacement = ("1" if rng.random() < 0.5 else "0") * encoding.bits_per_gene
    return chromosome[:start] + replacement + chromosome[end:]


def mutate(chromosome: str, method: MutationMethod, encoding: BinaryEncoding, rng: random.Random) -> str:
    if method == MutationMethod.BOUNDARY:
        return boundary(chromosome, encoding, rng)
    if method == MutationMethod.TWO_POINT:
        return two_point(chromosome, rng)
    return one_point(chromosome, rng)
