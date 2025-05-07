from abc import ABCMeta
import typing as t
from typing import Self, Any, overload
from collections.abc import Sequence, Mapping
from collections import Counter
from functools import cache, lru_cache
from numbers import Rational, Real, Integral
from fractions import Fraction as Q
import math
import warnings

from . import (
    DiatonicPitchBase,
    PitchWrapperBase,
    OPitchWrapper,
    PitchWrapper,
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
    _resolveStep,
    _resolveAcci,
)
from .utils.cls import cachedProp, cachedGetter, classProp, singleton
from .utils.number import alternateSignInts, primeFactors, pf2Rational

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

LOG_2_3 = math.log2(3)
LOG_2_3_M1 = math.log2(1.5)


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
    __slots__ = ("_pitchBaseType",)

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
def _getEDO(n: Integral) -> edo:
    n: int = int(n)
    fifthSize = round(n * LOG_2_3_M1)
    d = math.gcd(n, fifthSize)
    n //= d
    fifthSize //= d

    if n == 12:
        return edo12
    tuning = _createEdo(n)
    tuning.PitchBase._fifthSize = fifthSize
    return tuning


@cache
def _createEdo(n: int) -> edo:
    class _PitchBase(
        PitchWrapperBase["_OPitch", "_Pitch"], EDOPitchBase["_OPitch", "_Pitch"]
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

    class _OPitch(_PitchBase, OPitchWrapper["_Pitch"], OEDOPitch["_Pitch"]):
        __slots__ = ("_p", "_hash")

        def __new__(cls, *args, **kwargs):
            return cls._newImpl(OPitch(*args, **kwargs))

    class _Pitch(_PitchBase, PitchWrapper["_OPitch"], EDOPitch["_OPitch"]):
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


def co5_p2_p3(step: Integral, acci: Integral) -> tuple[int, int]:
    o, ostep = divmod(step, 7)
    p3_noAcci = ostep * 2 % 7
    if p3_noAcci == 6:
        p3_noAcci = -1
    p3 = int(p3_noAcci + acci * 7)
    p2 = -(p3_noAcci // 2 + p3_noAcci) - acci * 11 + o
    # an alternative way to get `p2`:
    # q, r = divmod(p3, 7)
    # p2 = -(r * 3 // 2) - q * 11 + o
    print(p3, p2, o, ostep)
    return p2, p3


def pythagoreanFreq(step: int, acci: int) -> Rational:
    p2, p3 = co5_p2_p3(step, acci)
    return pf2Rational(p2, p3)


def pythagoreanPos(step: int, acci: int) -> float:
    p2, p3 = co5_p2_p3(step, acci)
    return p2 + p3 * LOG_2_3


_FJS_TOL = Q(65, 63)


@lru_cache
def fjsMaster(n: Integral) -> tuple[int, int]:
    """
    Implementation of the FJS master algorithm.
    Assumes `n` to be a prime number greater than 3.
    The return value is a pair `(p2, p3)` where `2**p2 * 3**p3 * n` gives the **formal comma**
    defined in FJS.

    **See**: https://misotanni.github.io/fjs/en/crash.html#the-fjs-master-algorithm
    """

    neg_p2_prime = math.floor(math.log2(n))
    for co5 in alternateSignInts():
        neg_p2_co5 = math.floor(LOG_2_3 * co5)
        f1, f2 = pf2Rational(-neg_p2_co5, co5), Q(n, 1 << neg_p2_prime)
        if (f1 / f2 if f1 > f2 else f2 / f1) < _FJS_TOL:
            p2 = -neg_p2_prime + neg_p2_co5
            p3 = -co5
            # comma == f2 / f1 == 2**p2 * 3**p3 * freq
            return p2, p3


def _parseAdjust(src: str) -> Q:
    if src[0] == "/":
        adjust = Q(1, int(src[1:]))
    else:
        adjust = Q(src)
    if adjust == 0:
        raise ValueError("`adjust` cannot be zero.")
    elif adjust < 0:
        warnings.warn(
            "`adjust` should be a positive rational number. Here the absolute value is taken."
        )
        adjust = -adjust
    return adjust


def _adjust2Str(adjust: Q) -> str:
    if adjust == 1:
        adjustText = ""
    elif adjust.numerator == 1:
        adjustText = f"(/{adjust.denominator})"
    else:
        adjustText = f"({adjust!s})"
    return adjustText


def _commaPf(adjust: Q, target: Counter[int] = None) -> Counter[int]:
    if target is None:
        target = Counter()
    pf_adjust = primeFactors(adjust)
    for prime, power in pf_adjust.items():
        p2, p3 = fjsMaster(prime)
        target[prime] += power
        target[2] += p2 * power
        target[3] += p3 * power
    return target


def _resolveFreq(freq: Real | str | PitchNotationBase, limit=_MISSING) -> Q:
    if isinstance(freq, PitchNotationBase):
        freq = freq.freq
    freq = Q(freq)
    if freq < 0:
        raise ValueError("Frequency should not be zero.")
    elif freq < 0:
        freq = -freq
        warnings.warn("Frequency should be positive. Here the absolute value is taken.")
    if limit is not _MISSING:
        freq = freq.limit_denominator(limit)
    return freq


def _resolveAdjust(adjust: Rational | str | Mapping[int, int] = 1) -> Q:
    if isinstance(adjust, str):
        adjust = _parseAdjust(adjust)
    if adjust == 0:
        raise ValueError("`adjust` cannot be zero.")
    elif adjust < 0:
        warnings.warn(
            "`adjust` should be a positive rational number. Here the absolute value is taken."
        )
        adjust = -adjust
    if adjust != 1:
        pf = primeFactors(adjust)
        del pf[2], pf[3]
        adjust = pf2Rational(pf)
    return adjust


class JIPitchBase(PitchWrapperBase["OJIPitch", "JIPitch"]):
    @classProp
    def opitchType(self) -> type["OJIPitch"]:
        return OJIPitch

    @classProp
    def pitchType(self) -> type["JIPitch"]:
        return JIPitch

    @classmethod
    @lru_cache
    def _fromFreq(cls, freq: Q) -> Self:
        pf = primeFactors(freq)
        p2 = pf.pop(2, 0)
        p3 = pf.pop(3, 0)
        adjust = pf2Rational(pf)
        for prime, power in pf.copy().items():
            p2_fjs, p3_fjs = fjsMaster(prime)
            dp2, dp3 = p2_fjs * power, p3_fjs * power
            pf[2] += dp2
            pf[3] += dp3
            p2 -= dp2
            p3 -= dp3
        res = cls.opitchType._newHelper(OPitch.co5(p3), adjust)
        res._comma = pf2Rational(pf)  # cache
        q, r = divmod(p3, 7)
        neg_p2_expected = (r * 3 // 2) + q * 11
        pf[3] += p3
        pf[2] -= neg_p2_expected
        res._freq = pf2Rational(pf)  # cache
        if issubclass(cls, PitchNotation):
            o = p2 + neg_p2_expected
            res = cls._newHelper(res, o)
        return res

    @property
    def adjust(self) -> Q:
        return self.opitch.adjust

    @property
    def comma(self) -> Q:
        return self.opitch.comma

    @cachedProp
    def freq(self) -> Q:
        if self.adjust == 1:
            return pythagoreanFreq(self.step, self.acci)
        else:
            p2, p3 = co5_p2_p3(self.step, self.acci)
            if self.adjust == 1:
                return pf2Rational(p2, p3)
            pf_freq = _commaPf(self.adjust, Counter({2: p2, 3: p3}))
            return pf2Rational(pf_freq)

    def isEnharmonic(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return self.freq == other.freq
        return self._p == other._p and self.adjust == other.adjust

    def __eq__(self, other: Any):
        if not isinstance(other, self.__class__):
            return False
        return self._p == other._p and self.adjust == other.adjust


class OJIPitch(JIPitchBase, OPitchWrapper["JIPitch"]):
    __slots__ = ("_p", "_adjust", "_freq", "_comma", "_hash")
    _adjust: Q
    acci: int

    if t.TYPE_CHECKING:  # pragma: no cover

        @overload
        def __new__(
            cls,
            *,
            freq: Real | str | PitchNotationBase,
            limit: Integral | None = None,
        ) -> Self:
            """
            Creates an `OJIPitch` object from a frequency value relative to middle C, presumably
            rational. When `freq` is a floating point number, it is taken the exact value as an
            integer ratio. The `limit` parameter, if given, provides an upper bound for the
            denominator of the frequency value.
            """
            ...

        @overload
        def __new__(cls, src: str, /) -> Self:
            """
            Creates an `OJIPitch` object from a string notation, which follows mostly the same
            rule as the `OPitch` notation. However, the accidental is required to be an integer.
            Also, an adjust value, following the FJS just intonation notation rule, can be
            specified in parenthesis.
            """
            ...

        @overload
        def __new__(
            cls,
            src: str | DiatonicPitchBase | m21.pitch.Pitch | m21.interval.Interval,
            /,
            *,
            adjust: Rational | str | Mapping[int, int] = 1,
        ) -> Self: ...

        @overload
        def __new__(
            cls,
            step: int | str,
            acci: int | str = 0,
            /,
            *,
            adjust: Rational | str | Mapping[int, int] = 1,
        ) -> Self: ...

    def __new__(
        cls,
        arg1=_MISSING,
        arg2=_MISSING,
        /,
        *,
        adjust=_MISSING,
        freq=_MISSING,
        limit=_MISSING,
    ):
        if freq is not _MISSING:
            if arg1 is not _MISSING or arg2 is not _MISSING or adjust is not _MISSING:
                warnings.warn(
                    "Positional arguments and `adjust` are ignored when `freq` is specified."
                )
            freq = _resolveFreq(freq, limit)
            return cls._fromFreq(freq)

        if limit is not _MISSING:
            warnings.warn("`limit` only works when `freq` is specified.")

        if arg2 is _MISSING:
            if arg1 is _MISSING:
                raise ValueError(
                    "At least one positional argument or `freq` is required."
                )
            if adjust is _MISSING:
                if isinstance(arg1, str):
                    return cls._parse(arg1)
                elif isinstance(arg1, cls):
                    return arg1
                adjust = Q(1)

        p = OPitch(arg1, arg2)
        if not p.acci.is_integer():
            # a JI pitch must have an integer accidental
            # TODO)) probably convert non-integer accidentals to integer accidentals?
            raise ValueError("Just Intonation only supports integer accidentals.")
        adjust = _resolveAdjust(adjust)
        return cls._newImpl(p, adjust)

    @classmethod
    def _newImpl(cls, p: OPitch, adjust: Rational) -> Self:
        self = super()._newImpl(p)
        self._adjust = adjust
        return self

    @classmethod
    def _parse(cls, src: str) -> Self:
        adjustStart = src.find("(")
        if adjustStart < 0:
            p = OPitch._parse(src)
            if not p.acci.is_integer():
                raise ValueError("Just intonation only supports integer accidentals.")
            adjust = Q(1)
        else:
            p = OPitch._parse(src[:adjustStart])
            if not p.acci.is_integer():
                raise ValueError("Just intonation only supports integer accidentals.")
            if src[-1] != ")":
                raise ValueError(f"Invalid JI pitch notation: {src}.")
            adjust = _parseAdjust(src[adjustStart + 1 : -1])
        return cls._newHelper(p, adjust)

    @property
    def adjust(self) -> Q:
        return self._adjust

    @cachedProp
    def comma(self) -> Q:
        return pf2Rational(_commaPf(self.adjust))

    def __str__(self) -> str:
        return f"{self._p!s}{_adjust2Str(self.adjust)}@ji3"

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

    @cachedGetter
    def __hash__(self):
        return hash((self.__class__, self._p, self._adjust))


class JIPitch(JIPitchBase, PitchWrapper[OJIPitch]):
    __slots__ = ("_opitch", "_o", "_freq", "_hash")

    if t.TYPE_CHECKING:  # pragma: no cover
        # constructors inherited from `OJIPitch` but has an additional keyword argument `o`

        @overload
        def __new__(
            cls,
            /,
            *,
            freq: Real | str | PitchNotationBase,
            limit: Integral | None = None,
            o: Integral,
        ) -> Self: ...

        @overload
        def __new__(cls, src: str | OJIPitch, /, *, o: Integral) -> Self: ...

        @overload
        def __new__(
            cls,
            src: str | DiatonicPitchBase | m21.pitch.Pitch | m21.interval.Interval,
            /,
            *,
            adjust: Rational | str | Mapping[int, int] = 1,
            o: Integral,
        ) -> Self: ...

        @overload
        def __new__(
            cls,
            step: int | str,
            acci: int | str = 0,
            /,
            *,
            adjust: Rational | str | Mapping[int, int] = 1,
            o: Integral,
        ) -> Self: ...

        # new constructors for `JIPitch`

        @overload
        def __new__(
            cls,
            /,
            *,
            freq: Real | str | PitchNotationBase,
            limit: Integral | None = None,
        ) -> Self: ...

        @overload
        def __new__(cls, src: str, /) -> Self: ...

        @overload
        def __new__(
            cls,
            src: str | DiatonicPitchBase | m21.pitch.Pitch | m21.interval.Interval,
            /,
            *,
            adjust: Rational | str | Mapping[int, int] = 1,
        ) -> Self: ...

    def __new__(
        cls,
        arg1=_MISSING,
        arg2=_MISSING,
        /,
        *,
        o=_MISSING,
        adjust=_MISSING,
        freq=_MISSING,
        limit=_MISSING,
    ):
        if o is not _MISSING:
            opitch = OJIPitch(arg1, arg2, freq=freq, limit=limit, adjust=adjust)
            o = int(o)
            return cls._newHelper(opitch, o)

        if freq is not _MISSING:
            if arg1 is not _MISSING or arg2 is not _MISSING or adjust is not _MISSING:
                warnings.warn(
                    "Positional arguments and `adjust` are ignored when `freq` is specified."
                )
            freq = _resolveFreq(freq, limit)
            return cls._fromFreq(freq)

        if arg2 is _MISSING:
            if arg1 is _MISSING:
                raise ValueError(
                    "At least one positional argument or `freq` is required."
                )
            if adjust is _MISSING:
                if isinstance(arg1, str):
                    return cls._parse(arg1)
                elif isinstance(arg1, cls):
                    return arg1
                adjust = Q(1)

        step = _resolveStep(arg1)
        o, ostep = divmod(step, 7)
        acci = _resolveAcci(arg2)
        adjust = _resolveAdjust(adjust)
        opitch = OJIPitch._newHelper(OPitch._newHelper(ostep, acci), adjust)
        return cls._newHelper(opitch, o)

    @classmethod
    def _parse(cls, src: str) -> Self:
        if not src[-1].isdigit():
            opitch = OJIPitch._parse(src)
            o = 0
        else:
            opitch, o = src.rsplit("_", 1)
            opitch = OJIPitch._parse(opitch)
            o = int(o)
        return cls._newHelper(opitch, o)

    def __str__(self) -> str:
        return f"{self.opitch._p!s}{_adjust2Str(self.adjust)}_{self.o}@ji3"

    @cachedGetter
    def __hash__(self):
        return hash((self.__class__, self.opitch, self.o))


@singleton
class JustIntonation(Temperament[JIPitchBase, OJIPitch, JIPitch]):
    __slots__ = ()

    def __new__(cls) -> Self:
        return super().__new__(cls, JIPitchBase)


ji = JustIntonation()
