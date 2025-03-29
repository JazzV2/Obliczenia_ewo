from abc import ABC, abstractmethod

import numpy as np

from .data_types import Point

class TargetFunction(ABC):
    @abstractmethod
    def __call__(self, x: float | np.ndarray , y: float | np.ndarray) -> float | np.ndarray:
        pass

    @property
    @abstractmethod
    def global_minimum(self) -> Point:
        pass


class RastriginFunction(TargetFunction):
    """
    Rastrigin function (2D):
    f(x, y) = 20 + x^2 + y^2 - 10*(cos(2*pi*x) + cos(2*pi*y))
    """
    def __call__(self, x: float | np.ndarray , y: float | np.ndarray) -> float | np.ndarray:
        return 20 + (x ** 2) + (y ** 2) - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

    def global_minimum(self) -> Point:
        return 0.0, 0.0

class HypersphereFunction(TargetFunction):
    """
    Hypersphere function:
    f(x) = sum(x_i^2)
    """
    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return x ** 2 + y ** 2

    def global_minimum(self) -> Point:
        return 0.0, 0.0


class RosenbrockFunction(TargetFunction):
    """
    Rosenbrock function:
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2) for i in range(N-1)
    """
    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return 100 * (y - x ** 2) ** 2 + (x - 1) ** 2

    def global_minimum(self) -> Point:
        return 1.0, 1.0

class Styblinski_and_Tang(TargetFunction):
    # f(x)=0.5\sum_{i=0}^{N-1}(x_i^4-16x_i^2+x_i)
    # (-78.3323314075428, [-2.903534, -2.903534])

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return 0.5 * ((x ** 4 - 16 * x ** 2 + x) + (y ** 4 - 16 * y ** 2 + y))

    def global_minimum(self) -> Point:
        return -2.903534, -2.903534