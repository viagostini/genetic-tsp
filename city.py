from __future__ import annotations

import numpy as np


class City:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"<City ({self.x}, {self.y})>"

    @classmethod
    def random_city(cls):
        return City(x=np.random.uniform(0, 200), y=np.random.uniform(0, 200))

    def distance_to(self, other: City) -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
