from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

from .config import RangeConfig


@dataclass(frozen=True, slots=True)
class BinaryEncoding:
    range_config: RangeConfig
    bits_per_gene: int
    max_step: int

    @classmethod
    def build(cls, range_config: RangeConfig) -> "BinaryEncoding":
        steps = max(1, round((range_config.upper - range_config.lower) / range_config.precision))
        bits_per_gene = max(1, ceil(log2(steps + 1)))
        return cls(range_config=range_config, bits_per_gene=bits_per_gene, max_step=steps)

    @property
    def chromosome_length(self) -> int:
        return self.bits_per_gene * self.range_config.dimensions

    def encode_value(self, value: float) -> str:
        clipped = min(max(value, self.range_config.lower), self.range_config.upper)
        step = round((clipped - self.range_config.lower) / self.range_config.precision)
        step = min(max(step, 0), self.max_step)
        return format(step, f"0{self.bits_per_gene}b")

    def decode_value(self, bits: str) -> float:
        step = int(bits, 2)
        step = min(step, self.max_step)
        value = self.range_config.lower + step * self.range_config.precision
        return min(value, self.range_config.upper)

    def encode(self, values: list[float]) -> str:
        return "".join(self.encode_value(value) for value in values)

    def decode(self, chromosome: str) -> list[float]:
        values: list[float] = []
        size = self.bits_per_gene
        for index in range(0, len(chromosome), size):
            values.append(self.decode_value(chromosome[index:index + size]))
        return values
