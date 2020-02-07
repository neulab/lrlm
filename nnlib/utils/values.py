"""
Values and Counter Utilities
"""
import math
from typing import List

__all__ = ['Average', 'MovingAverage', 'WeightedAverage', 'SimpleAverage']


class Average:
    def add(self, value: float):
        raise NotImplementedError

    def value(self) -> float:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class MovingAverage(Average):
    def __init__(self, length: int, plateau_threshold: int = 5):
        # Assume optimization has reached plateau if moving average does not decrease for 5 consecutive iterations
        self.length = length
        self.values: List[float] = []
        self.sum = 0.0
        self.previous_best = math.inf
        self.plateau_iters = 0
        self.plateau_threshold = plateau_threshold

    def add(self, value: float):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.length:
            self.sum -= self.values.pop(0)

        val = self.value()
        if val < self.previous_best:
            self.previous_best = val
            self.plateau_iters = 0
        else:
            self.plateau_iters += 1

    def value(self) -> float:
        return float(self.sum) / min(len(self.values), self.length)

    def clear(self) -> None:
        self.values = []
        self.sum = 0
        self.previous_best = math.inf
        self.plateau_iters = 0

    def decreasing(self) -> bool:
        return self.plateau_iters <= self.plateau_threshold

    def reset_stats(self) -> None:
        self.plateau_iters = 0


class WeightedAverage(Average):
    def __init__(self):
        self.sum = 0.0
        self.count = 0.0

    def add(self, value: float, count: float = 1.0):
        self.sum += value * count
        self.count += count

    def value(self) -> float:
        return self.sum / self.count

    def clear(self) -> None:
        self.sum = 0.0
        self.count = 0.0


class SimpleAverage(Average):
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value: float):
        self.sum += value
        self.count += 1

    def value(self) -> float:
        return self.sum / self.count

    def clear(self) -> None:
        self.sum = 0.0
        self.count = 0
