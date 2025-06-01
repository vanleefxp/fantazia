from __future__ import annotations

from abc import abstractmethod
from typing import Self
from collections.abc import Sequence
from numbers import Real

import numpy as np

from . import base as _abc_base
from ...utils.cls import classProp, cachedClassProp
from ...math_.mos import IntMOSPattern

__all__ = [
    "Notation",
    "OPitch",
    "Pitch",
]


class Notation[OPType: "OPitch", PType: "Pitch"](_abc_base.Notation[OPType, PType]):
    @classProp
    @abstractmethod
    def pattern(cls) -> IntMOSPattern:
        raise NotImplementedError

    @classProp
    def edo(cls) -> int:
        return cls.pattern.edo

    @cachedClassProp
    def diatonic(cls) -> Sequence[int]:
        l_size, s_size, length = (
            cls.pattern.l_size,
            cls.pattern.s_size,
            len(cls.pattern),
        )
        return (
            np.cumsum(cls.pattern.steps) * (l_size - s_size)
            + np.arange(length) * s_size
        )

    @classProp
    def sharpness(self) -> int:
        return self.pattern.sharpness

    @property
    def step(self) -> int:
        raise NotImplementedError

    @property
    def acci(self) -> Real:
        raise NotImplementedError

    @property
    def tone(self) -> Real:
        return self.diatonic[self.step] + self.sharpness * self.acci


class OPitch[PType: "Pitch"](_abc_base.OPitch[PType], Notation[Self, PType]): ...


class Pitch[OPType: OPitch](_abc_base.Pitch[OPType], Notation[OPType, Self]): ...
