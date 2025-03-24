from enum import Enum, unique
from typing import List, Tuple


Point = Tuple[float, float]
Bounds: Point = (-5.12, 5.12)

Population = List[Point]


@unique
class Selection(Enum):
    ROULETTE = 'roulette'
    TOURNAMENT = 'tournament'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> list:
        return [s.value for s in cls]
