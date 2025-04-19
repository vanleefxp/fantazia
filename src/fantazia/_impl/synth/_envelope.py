from __future__ import annotations
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


@np.vectorize(excluded=(0, 2))
def _calcFading(self: Fading, t: float, tmax: float) -> float:
    t_in, t_out = self.fadeInTime, self.fadeOutTime
    if t_in + t_out < tmax:
        t_in = tmax * (t_in / (t_in + t_out))
        t_out = tmax - t_in
    if t < t_in:
        return self.fadeInEasing(t / t_in)
    elif t > tmax - t_out:
        return self.fadeOutEasing((tmax - t) / t_out)
    else:
        return 1


@np.vectorize(excluded=(0, 2))
def _calcADSR(self: ADSR, t: float, tmax: float) -> float:
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

    def __call__(self, t, tmax=np.inf):
        return _calcFading(self, t, tmax)


@dataclass(frozen=True)
class ADSR(Envelope):
    attackTime: float
    decayTime: float
    sustainLevel: float
    releaseTime: float
    attackEasing: Easing = linear
    decayEasing: Easing = linear
    releaseEasing: Easing = linear

    def __call__(self, t, tmax=np.inf):
        return _calcADSR(self, t, tmax)
