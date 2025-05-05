from abc import ABCMeta
import typing as t
from typing import Self, Any, overload
from collections.abc import Sequence, Mapping
from collections import Counter
from functools import cache, lru_cache
from numbers import Real, Rational
from fractions import Fraction as Q
import math

import numpy as np

from . import (
    DiatonicPitchBase,
    WrappedPitchBase,
    OWrappedPitch,
    WrappedPitch,
    EDOPitchBase,
    OEDOPitch,
    EDOPitch,
    PitchBase,
    OPitch,
    Pitch,
    PitchNotationBase,
    OPitchNotation,
    PitchNotation,
    _MISSING,
)
from .utils.cls import cachedProp, cachedGetter, classProp, singleton
from .utils.number import alternateSignInts, _primeFactors, primeFactors, pf2Rational

if t.TYPE_CHECKING:
    import music21 as m21

__all__ = [
    "edo",
    "edo12",
    "edo17",
    "edo19",
    "edo22",
    "edo31",
    "edo41",
    "edo53",
    "JustIntonation",
    "ji",
]

LOG_2_3 = np.log2(3)
LOG_2_3_M1 = np.log2(3 / 2)


@lru_cache
def co5Order(step: int, acci: int) -> int:
    r = step * 2 % 7
    if r == 6:
        r = -1
    return acci * 7 + r


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


@lru_cache
def _getEDO(n: int) -> edo:
    fifthSize = round(n * LOG_2_3_M1)
    d = np.gcd(n, fifthSize)
    n //= d
    fifthSize //= d

    if n == 12:
        return edo12
    tuning = _createEdo(n)
    tuning._fifthSize = fifthSize
    return tuning


@cache
def _createEdo(n: int) -> edo:
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
    return pf2Rational(p2, p3)


def pythagoreanPos(step: int, acci: int) -> float:
    p2, p3 = pythagoreanFreqP(step, acci)
    return p2 + p3 * LOG_2_3


_FJS_TOL = Q(65, 63)


@lru_cache
def fjsMaster(n: int) -> tuple[int, int]:
    """
    Implementation of the FJS master algorithm.
    Assumes `n` to be a prime number greater than 3.
    The return value is a pair `(p2, p3)` where `2**p2 * 3**p3 * n` gives the **formal comma**
    defined in FJS.

    **See**: https://misotanni.github.io/fjs/en/crash.html#the-fjs-master-algorithm
    """

    neg_p2_prime = math.floor(math.log2(n))
    for co5 in alternateSignInts():
        neg_p2_k = math.floor(LOG_2_3 * co5)
        f1, f2 = pf2Rational(-neg_p2_k, co5), Q(n, 1 << neg_p2_prime)
        if (f1 / f2 if f1 > f2 else f2 / f1) < _FJS_TOL:
            p2 = -neg_p2_prime + neg_p2_k
            p3 = -co5
            # comma == f2 / f1 == 2**p2 * 3**p3 * freq
            return p2, p3


class JIPitchBase(WrappedPitchBase["OJIPitch", "JIPitch"]):
    @classProp
    def opitchType(self) -> type["OJIPitch"]:
        return OJIPitch

    @classProp
    def pitchType(self) -> type["JIPitch"]:
        return JIPitch

    @cachedProp
    def freq(self) -> Real:
        if self.adjust == 1:
            return pythagoreanFreq(self.step, self.acci)
        else:
            p2, p3 = pythagoreanFreqP(self.step, self.acci)
            pf_freq = Counter({2: p2, 3: p3})

            pf_adjust = _primeFactors(self.adjust)
            for prime, power in pf_adjust.items():
                p2, p3 = fjsMaster(prime)
                pf_freq[prime] += power
                pf_freq[2] += p2 * power
                pf_freq[3] += p3 * power

            return pf2Rational(pf_freq)

    @property
    def pos(self) -> Real:
        return math.log2(self.freq)

        # if self.acci.is_integer():
        #     return pythagoreanFreq(self.step, self.acci)
        # return np.exp2(self._interpPos())

    @property
    def adjust(self) -> Rational:
        return self.opitch.adjust

    # def _interpPos(self) -> float:
    #     lower, t = divmod(self.acci, 1)
    #     lower = int(lower)
    #     lowerPos = pythagoreanPos(self.step, lower)
    #     upperPos = pythagoreanPos(self.step, lower + 1)
    #     return lowerPos * (1 - t) + upperPos * t

    # @property
    # def pos(self) -> float:
    #     if isinstance(self.acci, Integral):
    #         return pythagoreanPos(self.step, self.acci)
    #     return self._interpPos()

    def __str__(self):
        return f"{self._p!s}@ji3"

    def isEnharmonic(self, other: Any):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self == other

    def __eq__(self, other: Any):
        if not isinstance(other, self.__class__):
            return False
        return self._p == other._p and self.adjust == other.adjust

    @cachedGetter
    def __hash__(self):
        return hash((self.__class__, self._p, self._adjust))


class OJIPitch(JIPitchBase, OWrappedPitch["JIPitch"]):
    __slots__ = ("_p", "_adjust", "_freq", "_hash")

    if t.TYPE_CHECKING:  # pragma: no cover

        @overload
        def __new__(cls, *, freq: Rational) -> Self: ...

        @overload
        def __new__(
            cls,
            src: str | DiatonicPitchBase | m21.pitch.Pitch | m21.interval.Interval,
            *,
            adjust: Rational | str | Mapping[int, int] = 1,
        ) -> Self: ...

        @overload
        def __new__(
            cls,
            step: int | str,
            acci: int | str = 0,
            *,
            adjust: Rational | str | Mapping[int, int] = 1,
        ) -> Self: ...

    def __new__(cls, arg1=_MISSING, arg2=_MISSING, *, adjust=1, freq=_MISSING):
        if freq is not _MISSING:
            return cls._fromFreq(freq)
        p = OPitch(arg1, arg2)
        if isinstance(adjust, str):
            if adjust[0] == "/":
                adjust = Q(1, int(adjust[1:]))
            else:
                adjust = Q(adjust)
        if adjust != 1:
            pf = primeFactors(adjust)
            del pf[2], pf[3]
            adjust = pf2Rational(pf)
        return cls._newImpl(p, adjust)

    @classmethod
    @lru_cache
    def _newHelper(cls, p: OPitch, adjust: Rational = 1) -> Self:
        return cls._newImpl(p, adjust)

    @classmethod
    def _newImpl(cls, p: OPitch, adjust: Rational = 1) -> Self:
        self = super()._newImpl(p)
        self._adjust = adjust
        return self

    @classmethod
    @lru_cache
    def _fromFreq(cls, freq: Rational | str) -> Self:
        freq = Q(freq)
        pf = _primeFactors(freq)
        del pf[2]
        p3 = pf.pop(3, 0)
        adjust = pf2Rational(pf)
        for prime, power in pf.copy().items():
            p2_fjs, p3_fjs = fjsMaster(prime)
            dp2, dp3 = p2_fjs * power, p3_fjs * power
            pf[2] += dp2
            pf[3] += dp3
            p3 -= dp3

        res = cls._newHelper(OPitch.co5(p3), adjust)
        pf[3] += p3
        pf[2] -= math.floor(LOG_2_3 * p3)
        res._freq = pf2Rational(pf)
        return res

    @property
    def adjust(self) -> Rational:
        return self._adjust

    def __str__(self) -> str:
        if self.adjust == 1:
            return f"{self._p!s}@ji3"
        else:
            if self.adjust.numerator == 1:
                adjustText = f"/{self.adjust.denominator}"
            else:
                adjustText = str(self.adjust)
            return f"{self._p!s}({adjustText})@ji3"

    def __add__(self, other: Any) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        newP = self._p + other._p
        newAdjust = self.adjust * other.adjust
        return self._newHelper(newP, newAdjust)

    def __neg__(self) -> Self:
        newP = -self._p
        newAdjust = 1 / Q(self.adjust)
        return self._newHelper(newP, newAdjust)


class JIPitch(JIPitchBase, WrappedPitch[OJIPitch]):
    def __new__(cls, *args, **kwargs):
        return cls._newImpl(Pitch(*args, **kwargs))


@singleton
class JustIntonation(Temperament[JIPitchBase, OJIPitch, JIPitch]):
    __slots__ = ()

    def __new__(cls) -> Self:
        return super().__new__(cls, JIPitchBase)


ji = JustIntonation()
