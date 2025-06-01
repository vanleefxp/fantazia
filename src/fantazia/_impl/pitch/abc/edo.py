from abc import abstractmethod
from typing import Self, Any
from numbers import Real, Rational
from fractions import Fraction as Q
from collections.abc import Sequence

import pyrsistent as pyr

from . import quasiDiatonic as _abc_quasiDiatonic
from ...utils.cls import classProp
from ...utils.number import qdiv
from ...math_.ntheory import Monzo


class Notation[OPType: "OPitch", PType: "Pitch"](
    _abc_quasiDiatonic.Notation[OPType, PType]
):
    __slots__ = ()

    @classProp
    @abstractmethod
    def edo(cls) -> int:
        """
        Number of equal divisions of the octave.
        """
        raise NotImplementedError

    @classProp
    @abstractmethod
    def sharpness(self) -> Real:
        """
        Number of EDO steps a sharp sign raises.

        **See**: <https://en.xen.wiki/w/Sharpness>
        """
        raise NotImplementedError

    @classProp
    @abstractmethod
    def diatonic(cls) -> Sequence[int]:
        """
        A sequence denoting mapping from diatonic steps to EDO steps.
        """
        raise NotImplementedError

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


class OPitch[PType: "Pitch"](_abc_quasiDiatonic.OPitch[PType], Notation[Self, PType]):
    __slots__ = ()

    @classmethod
    def _fromStepAndTone(cls, step: int, tone: Real) -> Self:
        step, o = divmod(step, cls.n_steps)
        tone -= o * cls.edo
        acci = qdiv(tone - cls.diatonic[step], cls.sharpness)
        return cls._fromStepAndAcci(step, acci)

    def isEnharmonic(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return super().isEnharmonic(other)
        return self.tone % self.edo == other.tone % self.edo

    @property
    def tone(self) -> Real:
        return self.diatonic[self.step] + self.acci * self.sharpness


class Pitch[OPType: "OPitch"](_abc_quasiDiatonic.Pitch[OPType], Notation[OPType, Self]):
    __slots__ = ()

    @classmethod
    def _fromStepAndTone(cls, step: int, tone: Real) -> Self:
        ostep, o = divmod(step, cls.n_steps)
        tone -= o * cls.edo
        acci = qdiv(tone - cls.diatonic[ostep], cls.sharpness)
        return cls._fromOPitchAndO(cls.OPitch._fromStepAndAcci(ostep, acci), o)
