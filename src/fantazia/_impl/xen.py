from abc import ABCMeta, abstractmethod
from numbers import Real, Rational
from typing import Self
from functools import lru_cache
from fractions import Fraction as Q

import numpy as np

from . import OPitch, Pitch, PitchBase

__all__ = ["TuningSystem", "EDO", "Pythagorean"]

LOG_2_3 = np.log2(3)
LOG_2_3_M1 = np.log2(3 / 2)


@lru_cache
def co5Order(deg: int, acci: int) -> int:
    r = deg * 2 % 7
    if r == 6:
        r = -1
    return acci * 7 + r


class TuningSystem(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, pitch: PitchBase) -> Real:
        raise NotImplementedError

    def freq(self, pitch: PitchBase) -> Real:
        return 2 ** self(pitch)


class EDO(TuningSystem):
    """
    Equal temperament, or equal division of the octave.

    **See**: <https://en.xen.wiki/w/EDO>
    """

    __slots__ = ("_n", "_fifthSize")

    def __new__(cls, n: int = 12) -> Self:
        fifthSize = round(n * LOG_2_3_M1)
        d = np.gcd(n, fifthSize)
        n //= d
        fifthSize //= d
        self = cls._newHelper(n)
        self._fifthSize = fifthSize
        return self

    @classmethod
    @lru_cache
    def _newHelper(cls, n: int) -> Self:
        self = super().__new__(cls)
        self._n = n
        return self

    @property
    def n(self) -> int:
        return self._n

    @property
    def fifthSize(self) -> int:
        if not hasattr(self, "_fifthSize"):
            self._fifthSize = round(self.n * LOG_2_3_M1)
        return self._fifthSize

    def tone(self, pitch: PitchBase) -> float:
        if isinstance(pitch, OPitch):
            deg, acci = pitch.step, pitch.acci
        elif isinstance(pitch, Pitch):
            deg, acci = pitch.opitch.step, pitch.acci
        octave = pitch.tone // 12
        if (rem := acci % 1) == 0:
            order = co5Order(deg, acci)
            return self.fifthSize * order % self.n + octave * self.n
        else:  # interpolate
            lower = co5Order(deg, np.floor(acci))
            upper = co5Order(deg, np.ceil(acci))
            lowerTone = self.fifthSize * lower % self.n
            upperTone = self.fifthSize * upper % self.n
            return lowerTone * (1 - rem) + upperTone * rem + octave * self.n

    def __call__(self, pitch: PitchBase) -> float:
        return self.tone(pitch) / self.n

    def __str__(self):
        return f"{self.n}edo"


class Pythagorean(TuningSystem):
    __slots__ = ()
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def freq(self, pitch: PitchBase) -> Rational:
        octave = pitch.tone // 12
        order = co5Order(pitch.step, pitch.acci)
        numer = 3**order
        denom = 1 << (np.floor(order * LOG_2_3) - octave)
        return Q(numer, denom)

    def __call__(self, pitch: PitchBase) -> float:
        return np.log2(float(self.freq(pitch)))
