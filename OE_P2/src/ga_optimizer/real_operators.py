"""Real chromosomes; all operators return fresh immutable tuples.

Out-of-domain offspring are clipped. BLX-alpha-beta takes the better parent
first; engine ranks parents using the requested objective direction.
"""
from __future__ import annotations

import random
from .config import RealCrossoverMethod, RealMutationMethod


def clip(values, lower: float, upper: float) -> tuple[float, ...]:
    return tuple(min(upper, max(lower, x)) for x in values)


def real_crossover(first, second, method: RealCrossoverMethod, rng: random.Random,
                   lower: float, upper: float, alpha: float = 0.5, beta: float = 0.25):
    if len(first) != len(second) or not first:
        raise ValueError("Parents must have the same positive dimension.")
    if method == RealCrossoverMethod.LINEAR:
        candidates = [tuple((a+b)/2 for a,b in zip(first,second)),
                      tuple(1.5*a-0.5*b for a,b in zip(first,second)),
                      tuple(-0.5*a+1.5*b for a,b in zip(first,second))]
    elif method == RealCrossoverMethod.AVERAGE:
        child = tuple((a+b)/2 for a,b in zip(first,second))
        candidates = [child, child]
    elif method == RealCrossoverMethod.ARITHMETIC:
        weight = rng.random()  # one coefficient for the complete pair
        candidates = [tuple(weight*a+(1-weight)*b for a,b in zip(first,second)),
                      tuple((1-weight)*a+weight*b for a,b in zip(first,second))]
    elif method in (RealCrossoverMethod.BLX_ALPHA, RealCrossoverMethod.BLX_ALPHA_BETA):
        def child():
            result = []
            for a,b in zip(first,second):
                distance = abs(a-b)
                if method == RealCrossoverMethod.BLX_ALPHA:
                    lo,hi = min(a,b)-alpha*distance, max(a,b)+alpha*distance
                elif a <= b:  # a belongs to the better parent
                    lo,hi = a-alpha*distance, b+beta*distance
                else:
                    lo,hi = b-beta*distance, a+alpha*distance
                result.append(rng.uniform(lo,hi))
            return result
        candidates = [child(), child()]
    else:
        raise ValueError("Unknown real crossover.")
    return tuple(clip(child,lower,upper) for child in candidates)


def real_mutate(chromosome, method: RealMutationMethod, rng: random.Random,
                lower: float, upper: float, sigma: float = 0.05):
    values = list(chromosome)
    index = rng.randrange(len(values))
    if method == RealMutationMethod.UNIFORM:
        values[index] = rng.uniform(lower,upper)
    elif method == RealMutationMethod.GAUSSIAN:
        values[index] += rng.gauss(0, sigma*(upper-lower))
    else:
        raise ValueError("Unknown real mutation.")
    return clip(values,lower,upper)
