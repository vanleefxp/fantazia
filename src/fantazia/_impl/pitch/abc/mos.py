from __future__ import annotations

from abc import abstractmethod
from typing import Self
from collections.abc import Sequence

import numpy as np

from . import edo as _abc_edo
from ...utils.cls import classProp, cachedClassProp
from ...math_.mos import IntMOSPattern

__all__ = [
    "Notation",
    "OPitch",
    "Pitch",
]


class Notation[OPType: "OPitch", PType: "Pitch"](_abc_edo.Notation[OPType, PType]):
    @classProp
    @abstractmethod
    def pattern(cls) -> IntMOSPattern:
        raise NotImplementedError

    @classProp
    def edo(cls) -> int:
        return cls.pattern.edo

    @classProp
    def sharpness(self) -> int:
        return self.pattern.sharpness

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


class OPitch[PType: "Pitch"](_abc_edo.OPitch[PType], Notation[Self, PType]): ...


class Pitch[OPType: OPitch](_abc_edo.Pitch[OPType], Notation[OPType, Self]): ...
