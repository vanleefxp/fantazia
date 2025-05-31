from __future__ import annotations

from abc import abstractmethod
from numbers import Real
from typing import Literal, Self
import math

from ...math_ import AbelianElement
from ...utils.cls import NewHelperMixin, ClassPropMeta, classProp


class PitchNotationBase[
    OPType: "OPitchNotation"[PType],
    PType: "PitchNotation"[OPType],
](AbelianElement, metaclass=ClassPropMeta):
    __slots__ = ()

    @classProp
    @abstractmethod
    def OPitch(cls) -> type[OPType]:
        """The general pitch type linked to the current pitch notation."""
        raise NotImplementedError

    @classProp
    @abstractmethod
    def Pitch(cls) -> type[PType]:
        """The octave-specific pitch type linked to the current pitch notation."""
        raise NotImplementedError

    @classProp
    def oP(cls) -> type[OPType]:
        """
        Alias for `cls.OPitch`.
        """
        return cls.OPitch

    @classProp
    def P(cls) -> type[PType]:
        """
        Alias for `cls.Pitch`.
        """
        return cls.Pitch

    @property
    def pos(self) -> Real:
        """
        Relative position of the pitch measured in octaves. Equals to the base-2 logarithm of
        the frequency relative to middle C.

        **Implementation Note**: concrete subclasses of `PitchNotationBase` should override
        at least one of the `pos` or `freq` property. Otherwise, the default implementation
        will result in infinite recursion.
        """
        return math.log2(float(self.freq))

    @property
    def freq(self) -> Real:
        """
        Frequency of the pitch relative to middle C.
        """
        return 2 ** float(self.pos)

    @property
    def opos(self) -> Real:
        """
        Relative position of the pitch in the octave. Equals to the fractional part of `pos`.
        """
        return self.pos % 1

    @property
    @abstractmethod
    def opitch(self) -> OPType:
        raise NotImplementedError

    @property
    def o(self) -> int:
        return self.octave

    @property
    def octave(self) -> int:
        return int(self.pos // 1)

    def isEnharmonic(self, other: Self) -> bool:
        return self.pos == other.pos

    def hz(self, middleA: float = 440) -> float:
        """
        Returns the frequency of the pitch in hertz.
        """
        return middleA * 2 ** (self.pos - 0.75)

    def atOctave(self, octave: int = 0) -> PType:
        """
        Place the current pitch at the given octave.
        """
        return self.opitch.atOctave(octave)


class OPitchNotation[PType: "PitchNotation"](PitchNotationBase[Self, PType]):
    __slots__ = ()

    @property
    def opitch(self) -> Self:
        return self

    @property
    def o(self) -> int:
        return 0

    def isEnharmonic(self, other: Self) -> bool:
        return self.pos % 1 == other.pos % 1


class PitchNotation[OPType: OPitchNotation](
    PitchNotationBase[OPType, Self], NewHelperMixin
):
    __slots__ = ()

    @classmethod
    def _newImpl(cls, opitch: OPType, o: int) -> Self:
        self = super().__new__(cls)
        self._opitch = opitch
        self._o = o
        return self

    @classmethod
    def _fromOPitchAndO(cls, opitch: OPType, o: int) -> Self:
        return cls._newHelper(opitch, o)

    @property
    def opitch(self) -> OPitchNotation:
        return self._opitch

    @property
    def o(self) -> int:
        return self._o

    @property
    def sgn(self) -> Literal[-1, 0, 1]:
        return (self.pos > 0) - (self.pos < 0)
