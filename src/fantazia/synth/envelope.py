from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["Envelope", "NoEnvelope", "Fading", "ADSR"]

type Easing = Callable[[float], float]


def linear(x: float) -> float:
    return x


class Envelope(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, t: float, tmax: float = np.inf) -> float:
        raise NotImplementedError


class NoEnvelope(Envelope):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, t: float, tmax: float = np.inf) -> float:
        return 1


@dataclass(frozen=True)
class Fading(Envelope):
    fadeInTime: float
    fadeOutTime: float
    fadeInEasing: Easing = linear
    fadeOutEasing: Easing = linear

    def __post_init__(self):
        @np.vectorize
        def _calc(t: float, tmax: float = np.inf) -> float:
            tin, tout = self.fadeInTime, self.fadeOutTime
            if tin + tout < tmax:
                tin = tmax * (tin / (tin + tout))
                tout = tmax - tin
            if t < tin:
                return self.fadeInEasing(t / tin)
            elif t > tmax - tout:
                return self.fadeOutEasing((tmax - t) / tout)
            else:
                return 1

        object.__setattr__(self, "calc", _calc)


@dataclass(frozen=True)
class ADSR(Envelope):
    attackTime: float
    decayTime: float
    sustainLevel: float
    releaseTime: float
    attackEasing: Easing = linear
    decayEasing: Easing = linear
    releaseEasing: Easing = linear

    def __post_init__(self):
        @np.vectorize
        def _calc(t: float, tmax: float = np.inf) -> float:
            a, d, s, r = (
                self.attackTime,
                self.decayTime,
                self.sustainLevel,
                self.releaseTime,
            )
            if t < a:
                return self.attackEasing(t / a)
            elif t < a + d:
                return self.decayEasing(1 - t / d) * (1 - s) + s
            else:
                releaseStart = max(a + d, tmax - r)
                if t < releaseStart:
                    return s
                else:
                    return s * (1 - self.releaseEasing((t - releaseStart) / r))

        object.__setattr__(self, "calc", _calc)
