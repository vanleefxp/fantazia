from __future__ import annotations

from abc import abstractmethod
from fractions import Fraction as Q
from collections.abc import Sequence
from numbers import Real, Rational, Integral
from typing import Self, Any
import math

import numpy as np
import pyrsistent as pyr

from .diatonic import DiatonicPitch, ODiatonicPitch, DiatonicPitchBase
from ...utils.cls import classProp, cachedClassProp
from ...utils.number import qdiv
from ...math_ import Monzo


class EDOPitchBase[OPType: "OEDOPitch", PType: "EDOPitch"](
    DiatonicPitchBase[OPType, PType]
):
    __slots__ = ()

    @classProp
    @abstractmethod
    def edo(cls) -> int:
        """
        Number of equal divisions of the octave.
        """
        raise NotImplementedError

    @cachedClassProp
    def fifthSize(cls) -> int:
        """
        Number of EDO steps corresponding to a perfect fifth (P5) interval.
        """
        return round(math.log2(1.5) * cls.edo)

    @cachedClassProp
    def sharpness(cls) -> int:
        """
        Number of EDO steps a sharp sign raises.

        **See**: <https://en.xen.wiki/w/Sharpness>
        """
        return cls.fifthSize * 7 - cls.edo * 4

    @cachedClassProp
    def diatonic(cls) -> Sequence[Integral]:
        """
        A sequence denoting mapping from diatonic steps to EDO steps.
        """
        res = np.arange(-1, 6) * cls.fifthSize - cls.edo * (np.arange(-1, 6) // 2)
        res = res[np.arange(1, 14, 2) % 7]
        res.flags.writeable = False
        return res

    @property
    def tone(self) -> Real:
        return self.opitch.tone + self.o * self.edo

    @property
    def otone(self) -> Real:
        return self.tone % self.edo

    @property
    def pos(self) -> Real:
        tone = self.tone
        if isinstance(tone, Rational):
            return Q(self.tone, self.edo)
        return self.tone / self.edo

    @property
    def freq(self) -> Real:
        pos = self.pos
        if isinstance(pos, Rational):
            return Monzo._newHelper(pyr.pmap({2: pos}))
        return 2**pos

    def isEnharmonic(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return super().isEnharmonic(other)
        return self.tone == other.tone


class OEDOPitch[PType: "EDOPitch"](ODiatonicPitch[PType], EDOPitchBase[Self, PType]):
    __match_args__ = ("step", "acci", "tone")
    __slots__ = ()

    @classmethod
    def _fromStepAndTone(cls, step: int, tone: Real) -> Self:
        step, o = divmod(step, 7)
        tone -= o * cls.edo
        acci = qdiv(tone - cls.diatonic[step], cls.sharpness)
        return cls._fromStepAndAcci(step, acci)

    def isEnharmonic(self, other: Any):
        if not isinstance(other, self.__class__):
            return super().isEnharmonic(other)
        return self.tone % self.edo == other.tone % self.edo

    @property
    def tone(self) -> Real:
        return self.diatonic[self.step] + self.acci * self.sharpness


class EDOPitch[OPType: OEDOPitch](DiatonicPitch[OPType], EDOPitchBase[OPType, Self]):
    __match_args__ = ("opitch", "o", "step", "acci", "tone")
    __slots__ = ()

    @classmethod
    def _fromStepAndTone(cls, step: int, tone: Real) -> Self:
        ostep, o = divmod(step, 7)
        tone -= o * cls.edo
        acci = qdiv(tone - cls.diatonic[ostep], cls.sharpness)
        return cls._fromOPitchAndO(cls.OPitch._fromStepAndAcci(ostep, acci), o)
