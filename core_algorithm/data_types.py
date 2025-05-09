from enum import Enum, unique
from typing import List, Tuple


Point = Tuple[float, float]
Bounds: Point = (-5.12, 5.12)

Population = List[Point]


class StreamlitEnum(Enum):
    """Helper class to safety keep values from Streamlit selectboxes in Enum format"""
    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> list:
        return [s for s in cls]


@unique
class FunctionBox(StreamlitEnum):
    RASTRIGIN = 'Rastrigin'
    HYPERSPHERE = 'Hypersphere'
    ROSENBROCK = 'Rosenbrock'
    Styblinski_and_Tang = "Styblinski and Tang"


@unique
class SelectionBox(StreamlitEnum):
    ROULETTE = 'roulette'
    TOURNAMENT = 'tournament'
    THEBEST = 'the best'


@unique
class CrossMethodBox(StreamlitEnum):
    ONEPOINT = 'One point'
    TWOPOINTS = 'Two points'
    HOMOGENEOUS = 'Homogeneous'
    GRAIN = 'Grain'
    ARTHMETIC = 'Arthmetic'
    TYPEALPHAMIX = 'Type alpha mix'
    TYPEALPHABETAMIX = 'Type alpha and beta mix'
    AVERAGE = 'Average'
    LINEAR = 'Linear'


@unique
class MutationMethodBox(StreamlitEnum):
    EDGE = 'Edge'
    ONEPOINT = 'One point'
    TWOPOINTS = 'Two points'
    INVERSION = 'Inversion'
    ISOSCELES = 'Isosceles'
    GAUSSIAN = 'Gaussian'
