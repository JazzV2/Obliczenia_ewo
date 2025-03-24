from typing import Union

import numpy as np


def rastrigin_function(x: Union[float, np.ndarray] , y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Rastrigin function (2D):
    f(x, y) = 20 + x^2 + y^2 - 10*(cos(2*pi*x) + cos(2*pi*y))
    """
    return 20 + (x ** 2) + (y ** 2) - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
