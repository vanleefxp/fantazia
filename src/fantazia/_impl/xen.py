from abc import ABCMeta
from typing import Self, Any
from collections.abc import Callable, Sequence
from functools import cache, lru_cache
from numbers import Real, Rational, Integral
from fractions import Fraction as Q

import numpy as np

from . import (
    DiatonicPitchBase,
    ODiatonicPitch,
    DiatonicPitch,
    EDOPitchBase,
    OEDOPitch,
    EDOPitch,
    PitchBase,
    OPitch,
    Pitch,
    PitchNotationBase,
    OPitchNotation,
    PitchNotation,
)
from .._impl.utils.cls import cachedProp, cachedGetter, classProp, singleton

__all__ = [
    "edo",
    "edo12",
    "edo17",
    "edo19",
    "edo22",
    "edo31",
    "edo41",
    "edo53",
    "Pythagorean",
    "ji3",
]

LOG_2_3 = np.log2(3)
LOG_2_3_M1 = np.log2(3 / 2)


@lru_cache
def co5Order(step: int, acci: int) -> int:
    r = step * 2 % 7
    if r == 6:
        r = -1
    return acci * 7 + r


class WrappedPitchBase[OPType: "OWrappedPitch", PType: "WrappedPitch"](
    DiatonicPitchBase[OPType, PType]
):
    """Helper type for wrapping a `PitchBase` with a different type."""

    @classmethod
    @lru_cache
    def _newHelper(cls, p: PitchBase) -> Self:
        return cls._newImpl(p)

    @classmethod
    def _newImpl(cls, p: PitchBase) -> Self:
        self = super().__new__(cls)
        self._p = p
        return self

    @property
    def step(self) -> int:
        return self._p.step

    @property
    def acci(self) -> Real:
        return self._p.acci

    @cachedProp
    def opitch(self) -> OPType:
        return self.opitchType._newHelper(self._p.opitch)

    def __add__(self, other: Any) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._newHelper(self._p + other._p)

    def __neg__(self) -> Self:
        return self._newHelper(-self._p)

    def __eq__(self, value) -> bool:
        return isinstance(value, self.__class__) and self._p == value._p

    @cachedGetter
    def __hash__(self) -> int:
        return hash((self.__class__, self._p))

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[Any, ...]]:
        return (self._newHelper, (self._p,))


class OWrappedPitch[PType: "WrappedPitch"](
    WrappedPitchBase[Self, PType], ODiatonicPitch[PType]
):
    @classmethod
    def _newImpl(cls, p: OPitch) -> Self:
        self = super().__new__(cls)
        self._p = p
        return self

    @classmethod
    def _fromStepAndAcci(cls, step: int, acci: Real) -> Self:
        return cls._newHelper(OPitch._fromStepAndAcci(step, acci))

    @property
    def opitch(self) -> Self:
        return self

    @property
    def o(self) -> int:
        return 0


class WrappedPitch[OPType: OWrappedPitch](
    WrappedPitchBase[OPType, Self], DiatonicPitch[OPType]
):
    @classmethod
    def _newImpl(cls, p: Pitch) -> Self:
        self = super().__new__(cls)
        self._p = p
        return self

    @property
    def o(self) -> int:
        return self._p.o


class Temperament[
    PBType: PitchNotationBase[OPType, PType],
    OPType: OPitchNotation[PType],
    PType: PitchNotation[OPType],
](metaclass=ABCMeta):
    __slots__ = "_pitchBaseType"

    def __new__(cls, pitchBaseType: type[PBType]):
        self = super().__new__(cls)
        self._pitchBaseType = pitchBaseType
        return self

    @property
    def PitchBase(self) -> type[PBType]:
        return self._pitchBaseType

    @property
    def OPitch(self) -> type[OPType]:
        return self.PitchBase.opitchType

    @property
    def Pitch(self) -> type[PType]:
        return self.PitchBase.pitchType

    @property
    def oP(self) -> type[OPType]:
        return self.OPitch

    @property
    def P(self) -> type[PType]:
        return self.Pitch


class edo[
    PBType: EDOPitchBase[OPType, PType],
    OPType: OEDOPitch[PType],
    PType: EDOPitch[OPType],
](Temperament[PBType, OPType, PType]):
    __slots__ = (
        "_edo",
        "_fifthSize",
        "_sharpness",
        "_diatonic",
        "_pitchBaseType",
        "_oPitchType",
        "_pitchType",
    )

    def __new__(cls, n: int = 12) -> Self:
        return _getEDO(n)

    @property
    def edo(self) -> int:
        """Number of equal divisions of the octave."""
        return self.PitchBase.edo

    @property
    def fifthSize(self) -> int:
        """The size of a fifth interval measured by edo steps."""
        return self.PitchBase.fifthSize

    @property
    def sharpness(self) -> int:
        return self.PitchBase.sharpness

    @property
    def diatonic(self) -> Sequence[int]:
        return self.PitchBase.diatonic

    @property
    def PitchBase(self) -> type[PBType]:
        """The type equivalent to `PitchBase` for the standard 12-EDO."""
        return self._pitchBaseType

    @property
    def OPitch(self) -> type[OPType]:
        return self._pitchBaseType.opitchType

    @property
    def Pitch(self) -> type[PType]:
        return self._pitchBaseType.pitchType


edo12 = Temperament.__new__(edo, PitchBase)
"""
**12 EDO**, or 12 tone equal temperament, which is the standard tuning system.

`edo12.PitchBase`, `edo12.OPitch` and `edo12.Pitch` will be redirected to `PitchBase`, `OPitch` 
and `Pitch` in the root package of `fantazia` respectively.
"""


@cache
def _getEDO(n: int) -> edo:
    fifthSize = round(n * LOG_2_3_M1)
    d = np.gcd(n, fifthSize)
    n //= d
    fifthSize //= d

    if n == 12:
        return edo12

    class _PitchBase(
        WrappedPitchBase["_OPitch", "_Pitch"], EDOPitchBase["_OPitch", "_Pitch"]
    ):
        @classProp
        def edo(cls) -> int:
            return n

        @classProp
        def opitchType(self) -> type["_OPitch"]:
            return _OPitch

        @classProp
        def pitchType(self) -> type["_Pitch"]:
            return _Pitch

        def __str__(self):
            return f"{self._p!s}@edo{self.edo}"

    _PitchBase._fifthSize = fifthSize

    class _OPitch(_PitchBase, OWrappedPitch["_Pitch"], OEDOPitch["_Pitch"]):
        __slots__ = ("_p", "_hash")

        def __new__(cls, *args, **kwargs):
            return cls._newImpl(OPitch(*args, **kwargs))

    class _Pitch(_PitchBase, WrappedPitch["_OPitch"], EDOPitch["_OPitch"]):
        __slots__ = ("_p", "_hash", "_opitch")

        def __new__(cls, *args, **kwargs):
            return cls._newImpl(Pitch(*args, **kwargs))

    _PitchBase.__name__ = f"Pitch{n}Base"
    _OPitch.__name__ = f"OPitch{n}"
    _Pitch.__name__ = f"Pitch{n}"

    return Temperament.__new__(edo, _PitchBase)


edo17 = _getEDO(17)
edo19 = _getEDO(19)
edo22 = _getEDO(22)
edo31 = _getEDO(31)
edo41 = _getEDO(41)
edo53 = _getEDO(53)


def pythagoreanFreqP(step: int, acci: int) -> tuple[int, int]:
    o, ostep = divmod(step, 7)
    co5Order = ostep * 2 % 7
    if co5Order == 6:
        co5Order = -1
    p2 = -(co5Order // 2 + co5Order) - acci * 11 + o
    p3 = co5Order + acci * 7
    return p2, p3


def pythagoreanFreq(step: int, acci: int) -> Rational:
    p2, p3 = pythagoreanFreqP(step, acci)
    if p2 > 0:
        if p3 > 0:
            return (1 << p2) * (3**p3)
        else:
            return Q(1 << p2, 3**-p3)
    else:
        if p3 > 0:
            return Q(3**p3, 1 << -p2)
        else:
            return Q(1, (1 << -p2) * (3**-p3))


def pythagoreanPos(step: int, acci: int) -> float:
    p2, p3 = pythagoreanFreqP(step, acci)
    return p2 + p3 * LOG_2_3


class PythagoreanPitchBase(WrappedPitchBase["OPythagoreanPitch", "PythagoreanPitch"]):
    @classProp
    def opitchType(self) -> type["OPythagoreanPitch"]:
        return OPythagoreanPitch

    @classProp
    def pitchType(self) -> type["PythagoreanPitch"]:
        return PythagoreanPitch

    @property
    def freq(self) -> Real:
        if self.acci.is_integer():
            return pythagoreanFreq(self.step, self.acci)
        return np.exp2(self._interpPos())

    def _interpPos(self) -> float:
        lower, t = divmod(self.acci, 1)
        lower = int(lower)
        lowerPos = pythagoreanPos(self.step, lower)
        upperPos = pythagoreanPos(self.step, lower + 1)
        return lowerPos * (1 - t) + upperPos * t

    @property
    def pos(self) -> float:
        if isinstance(self.acci, Integral):
            return pythagoreanPos(self.step, self.acci)
        return self._interpPos()

    def __str__(self):
        return f"{self._p!s}@ji3"

    def isEnharmonic(self, other: Any):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self == other


class OPythagoreanPitch(PythagoreanPitchBase, OWrappedPitch["PythagoreanPitch"]):
    def __new__(cls, *args, **kwargs):
        return cls._newImpl(OPitch(*args, **kwargs))


class PythagoreanPitch(PythagoreanPitchBase, WrappedPitch[OPythagoreanPitch]):
    def __new__(cls, *args, **kwargs):
        return cls._newImpl(Pitch(*args, **kwargs))


@singleton
class Pythagorean(
    Temperament[PythagoreanPitchBase, OPythagoreanPitch, PythagoreanPitch]
):
    __slots__ = ()

    def __new__(cls) -> Self:
        return super().__new__(cls, PythagoreanPitchBase)


ji3 = Pythagorean()
