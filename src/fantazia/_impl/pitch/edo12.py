from __future__ import annotations

from typing import overload, Literal, Any, Self
import typing as t
from collections.abc import Sequence, Callable
from functools import lru_cache
from numbers import Real, Integral
import importlib.util
from bisect import bisect_left, bisect_right
from fractions import Fraction as Q
import re
import warnings

from .abc import base as _abc_base, diatonic as _abc_diatonic, edoCo5 as _abc_edo
from .abc.diatonic import (
    MAJOR_SCALE_TONES,
    STEP_NAMES,
    _qual2Str_int,
    _qualMap,
    _qualInvMap,
    _intervalQualityMap,
    _solfegeNamesInvMap,
)
from ..utils.cls import (
    cachedGetter,
    abstractmethod,
    classProp,
    cachedClassProp,
    noInstance,
    lazyIsInstance,
)
from ..utils.number import RMode, rdiv, rdivmod, clamp, rbisect, resolveInt
from ..utils.collection import cycGet

if t.TYPE_CHECKING:  # pragma: no cover
    import music21 as m21  # type: ignore

__all__ = [
    "Notation",
    "OPitch",
    "Pitch",
    "oP",
    "P",
    "ostep",
    "step",
    "acci",
    "AcciPref",
    "AcciPrefs",
]

_MISSING = object()
_leadingAlphaRe = re.compile(r"^(?:[a-z]+|[1-7])", re.IGNORECASE)
_trailingDigitsRe = re.compile(r"\d+$")
_multipleAugRe = re.compile(r"^A+$")
_multipleAugNumRe = re.compile(r"^\[A\*(\d+)\]")
_multipleDimRe = re.compile(r"^d+$")
_multipleDimNumRe = re.compile(r"^\[d\*(\d+)\]")

type AcciPref = Callable[[Real], int]
"""
Type alias for a function that takes a tone in half-steps and returns the preferred step for
that tone. The accidental can be later computed by taking the difference between the given
tone and the standard reference tone in C major scale. Some predefined accidental preference
rules can be found in `AcciPrefs`.
"""


def _parseMicrotone(src: str) -> int | Q | float:
    if len(src) == 0:
        return 0
    if src[0] != "[" or src[-1] != "]":
        raise ValueError(f"Invalid microtone format: {src}")
    valueSrc = src[1:-1]  # extract content inside the brackets
    if "/" in valueSrc:  # fractional value
        return Q(valueSrc)
    else:  # float value
        return float(valueSrc)
    # TODO)) consider support for more complicated mathematic expressions
    # this may cause security problems is `sympy.S` is directly used


def _parseAcci(src: str) -> int | Q | float:
    if len(src) == 0:  # no accidental (natural)
        return 0
    if src[0] == "[":  # numeric accidental value wrapped in []
        return _parseMicrotone(src)
    else:
        acci = 0
        for ch in src:
            match ch:
                case "+":  # "+" for sharp
                    acci += 1
                case "-":  # "-" for flat
                    acci -= 1
                case _:
                    raise ValueError(f"Invalid accidental token: {ch}")
        return acci


def _parseOStep(src: str) -> int:
    if len(src) == 1:  # single letter or number note name
        if src.isdigit():  # a number from 1 to 7
            step = int(src) - 1
            if step < 0 or step >= 7:
                raise ValueError(f"Invalid step symbol: {src}")
            return step
        else:  # a letter note name in CDEFGAB
            src = src.upper()
            step = ord(src) - 67
            if step < -2 or step > 4:
                raise ValueError(f"Invalid step symbol: {src}")
            if step < 0:
                step += 7
            return step
    else:  # solfege name
        src = src.lower()
        step = _solfegeNamesInvMap.get(src)
        if step is None:
            raise ValueError(f"Invalid step symbol: {src}")
        return step


def _parseStep(src: str) -> int:
    if not src[-1].isdigit():
        # octave not specified, assume octave 0
        return _parseOStep(src)
    else:
        ostep, octave = src.rsplit("_", 1)
        ostep = _parseOStep(ostep)
        return ostep + 7 * octave


def _parseQualBase(src: str) -> int:
    quality = _intervalQualityMap.get(src)
    if quality is None:
        if _multipleAugRe.match(src):
            return len(src) + 1
        elif match := _multipleAugNumRe.match(src):
            n = int(match.group(1))
            return n + 1
        elif _multipleDimRe.match(src):
            return -len(src) - 1
        elif match := _multipleDimNumRe.match(src):
            n = int(match.group(1))
            return -n - 1
        raise ValueError(f"Invalid interval quality: {src}")
    return quality


def _parseQual(src: str, step: int) -> Real:
    if src[0] == "[":
        end = src.find("]")
        if end < 0:
            raise ValueError(f"Invalid interval quality: {src}")
        end += 1
        qualitySrc = src[:end]
        microtoneSuffix = src[end:]
    else:
        start = src.find("[")
        if start < 0:
            qualitySrc = src
            microtoneSuffix = ""
        else:
            qualitySrc = src[:start]
            microtoneSuffix = src[start:]

    qualBase = _parseQualBase(qualitySrc)
    microtone = _parseMicrotone(microtoneSuffix)
    acci = _qualInvMap(step, qualBase) + microtone
    qual = _qualMap(step, acci)
    return qual


def _parseInterval(src: str) -> tuple[int, Real, bool]:
    neg = False
    if src[0] == "-":
        neg = True
        src = src[1:]
    elif src[0] == "+":
        src = src[1:]
    trailingNumMatch: re.Match[str] = _trailingDigitsRe.search(src)
    if trailingNumMatch is None:
        raise ValueError(f"Invalid interval format: {src}")
    end = trailingNumMatch.start()
    step = int(src[end:])
    if step == 0:
        raise ValueError("Interval number cannot be zero.")
    step = int(step) - 1
    qual = _parseQual(src[:end], step)
    acci = _qualInvMap(step, qual)
    return step, acci, neg


def _parseOPitch(src: str) -> tuple[int, Real]:
    # separate step and accidental
    stepNameMatch: re.Match[str] = _leadingAlphaRe.search(src)
    if stepNameMatch is None:
        raise ValueError(f"Invalid pitch format: {src}")
    stepName = stepNameMatch.group().lower()
    acciSrc = src[stepNameMatch.end() :]
    step = _parseOStep(stepName)
    acci = _parseAcci(acciSrc)

    return step, acci


def ostep(src: Integral | str) -> int:
    """Resolves an octave step representation to an integer between 0 and 6."""
    if isinstance(src, str):
        return _parseOStep(src)
    else:
        return int(src) % 7


def step(src: Integral | str) -> int:
    """Resolves a step representation in specific octave to an integer."""
    if isinstance(src, str):
        return _parseStep(src)
    else:
        return int(src)


def acci(src: Real | str) -> Real:
    """Resolves an accidental representation to a semitone value."""
    if isinstance(src, str):
        return _parseAcci(src)
    else:
        return resolveInt(src)


# aliases to avoid name conflict
_resolveStep = step
_resolveAcci = acci


def _resolveTone(src: Notation | Real) -> Real:
    if isinstance(src, Notation):
        return src.tone
    else:
        return resolveInt(src)


@noInstance
class AcciPrefs:
    """See `AcciPref` for details."""

    @staticmethod
    def SHARP(tone: Real) -> int:
        """
        Always use the lower step and sharp sign when a tone is not a standard tone in C major
        scale, both for standard 12edo pitches and microtonal pitches.

        Examples:

        | input | output | preferred name |
        |:-|:-|:-|
        | `1` | `0` | `C+` | C sharp |
        | `3` | `1` | `D+` | D sharp |
        | `Q("11/2")` | `3` | `F[+1/2]` | F quarter-tone-sharp |
        | `6` | `3` | `F+` | F sharp |
        | `Q("13/2")` | `3` | `F[+3/2]` | F 3-quarter-tones-sharp |
        | `8` | `4` | `G+` | G sharp |
        | `10` | `5` | `A+` | A sharp |
        """
        return bisect_right(MAJOR_SCALE_TONES, tone) - 1

    @staticmethod
    def FLAT(tone: Real) -> int:
        """
        Always use the upper step and flat sign when a tone is not a standard tone in C major
        scale,both for standard 12edo pitches and microtonal pitches.

        Examples:

        | input | output | preferred name |
        |:-|:-|:-|
        | `1` | `1` | `D-` | D flat |
        | `3` | `2` | `E-` | E flat |
        | `Q("11/2")` | `4` | `G[-3/2]` | G 3-quarter-tones-flat |
        | `6` | `4` | `G-` | G flat |
        | `Q("13/2")` | `4` | `G[-1/2]` | G quarter-tone-flat |
        | `8` | `5` | `A-` | A flat |
        | `10` | `6` | `B-` | B flat |
        """
        return bisect_left(MAJOR_SCALE_TONES, tone)

    @staticmethod
    def CLOSEST_SHARP(tone: Real) -> int:
        """
        For 12edo pitches, use the lower step and sharp sign when the tone is not a standard tone
        in C major scale. For microtonal pitches, choose the closest standard tone in C major scale.

        Examples:

        | input | output | preferred name |
        |:-|:-|:-|
        | `1` | `0` | `C+` | C sharp |
        | `3` | `1` | `D+` | D sharp |
        | `Q("11/2")` | `3` | `F[+1/2]` | F quarter-tone-sharp |
        | `6` | `3` | `F+` | F sharp |
        | `Q("13/2")` | `4` | `G[-1/2]` | G quarter-tone-flat |
        | `8` | `4` | `G+` | G sharp |
        | `10` | `5` | `A+` | A sharp |
        """
        if tone > 11.5:
            return 7
        return rbisect(MAJOR_SCALE_TONES, tone, rmode="f")

    @staticmethod
    def CLOSEST_FLAT(tone: Real) -> int:
        """
        For 12edo pitches, use the upper step and flat sign when the tone is not a standard tone
        in C major scale. For microtonal pitches, choose the closest standard tone in C major scale.

        Examples:

        | input | output | preferred name |
        |:-|:-|:-|
        | `1` | `1` | `D-` | D flat |
        | `3` | `2` | `E-` | E flat |
        | `Q("11/2")` | `3` | `F[+1/2]` | F quarter-tone-sharp |
        | `6` | `4` | `G-` | G flat |
        | `Q("13/2")` | `4` | `G[-1/2]` | G quarter-tone-flat |
        | `8` | `5` | `A-` | A flat |
        | `10` | `6` | `B-` | B flat |
        """
        if tone >= 11.5:
            return 7
        return rbisect(MAJOR_SCALE_TONES, tone, rmode="c")

    @staticmethod
    def CLOSEST_FLAT_F_SHARP(tone: Real) -> int:
        """
        Same as `CLOSEST_FLAT`, but for the tritone (`tone == 6`) case, use the F sharp instead
        of G flat. This is the default accidental preference rule for `OPitch.fromTone()`.

        Examples:

        | input | output | preferred name |
        |:-|:-|:-|
        | `1` | `1` | `D-` | D flat |
        | `3` | `2` | `E-` | E flat |
        | `Q("11/2")` | `3` | `F[+1/2]` | F quarter-tone-sharp |
        | `6` | `3` | `F+` | F sharp |
        | `Q("13/2")` | `4` | `G[-1/2]` | G quarter-tone-flat |
        | `8` | `5` | `A-` | A flat |
        | `10` | `6` | `B-` | B flat |
        """
        if tone >= 11.5:
            return 7
        step = rbisect(MAJOR_SCALE_TONES, tone, rmode="c")
        if tone == 6:
            return 3
        else:
            return step


class Notation(_abc_edo.Notation["OPitch", "Pitch"]):
    """
    **12 EDO**, or 12 tone equal temperament, which is the standard tuning system.
    """

    __slots__ = ("_hash",)

    @classProp
    def edo(self) -> int:
        return 12

    @classProp
    def fifthSize(self) -> int:
        return 7

    @classProp
    def sharpness(self) -> int:
        return 1

    @classProp
    def diatonic(self) -> Sequence[int]:
        return MAJOR_SCALE_TONES

    @classProp
    def OPitch(self) -> type["OPitch"]:
        return OPitch

    @classProp
    def Pitch(self) -> type["Pitch"]:
        return Pitch

    if t.TYPE_CHECKING:  # pragma: no cover

        @overload
        def m21(
            self,
            *,
            keepNatural: bool = False,
            useQuartertone: bool = True,
            roundToQuartertone: bool = True,
            round: bool = True,
            rmode: RMode | str = RMode.E,
            asInterval: Literal[False] = False,
        ) -> m21.pitch.Pitch: ...

        @overload
        def m21(
            self,
            *,
            round: bool = True,
            rmode: RMode | str = RMode.D,
            asInterval: Literal[True] = True,
        ) -> m21.interval.Interval: ...

    @classmethod
    @abstractmethod
    def _parsePitch(self, src: str) -> Self:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _parseInterval(self, src: str) -> Self:
        raise NotImplementedError

    @classmethod
    @lru_cache
    def _parse(cls, src: str) -> Self:
        try:  # assume to be a pitch notation
            return cls._parsePitch(src)
        except ValueError:
            # print("not a pitch notation, trying an interval notation...")
            try:  # assume to be an interval notation
                return cls._parseInterval(src)
            except ValueError:
                raise ValueError(f"invalid pitch or interval string: {src}")

    @classmethod
    @abstractmethod
    def _fromM21Pitch(cls, m21_obj: m21.pitch.Pitch) -> Self:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _fromM21Interval(cls, m21_obj: m21.pitch.Pitch) -> Self:
        raise NotImplementedError

    @property
    def step(self) -> int:
        """
        Step of the pitch in C major scale, not considering accidentals. Equals to the step
        of octave pitch plus nominal octave times 7.
        """
        return self.opitch.step + self.o * 7

    @property
    @lru_cache
    def tone(self) -> Real:
        octave, ostep = divmod(self.step, 7)
        return int(MAJOR_SCALE_TONES[ostep]) + self.acci + octave * 12

    @property
    def otone(self) -> Real:
        return self.tone % 12

    @property
    def o(self) -> int:
        """
        Returns the nominal octave of the pitch, determined solely by step.

        e.g. `Pitch("B+_0").o` and `Pitch("C-_0").o` are both `0`.
        """
        return self.step // 7

    @property
    def octave(self) -> int:
        """
        Returns the actual octave of the pitch, determined by the tone instead of step.

        e.g. `Pitch("B+_0").octave` is `1` and `Pitch("C-_0").octave` is `-1`.
        """
        return int(self.tone // 12)

    # @property
    # def freq(self) -> float:
    #     """
    #     Frequency of the pitch relative to middle C.
    #     """
    #     return np.pow(2, self.tone / 12)

    @abstractmethod
    def withAcci(self, acci: Real = 0) -> Self:
        """
        Returns a new instance of the same class with the given accidental.
        """
        raise NotImplementedError

    def alter(self, acci: Real = 0) -> Self:
        """Alter the accidental of the pitch by the given amount."""
        return self.withAcci(self.acci + acci)

    def atOctave(self, octave: int = 0) -> Pitch:
        """
        Place the current pitch at the given octave.

        **Note**: Here octave is the actual octave, not the nominal octave. To place a pitch `p`
        at a nominal octave `o`, call `fz.Pitch(p, o)` instead.
        """
        return super().atOctave(octave)

    def isEnharmonic(self, other: Self) -> bool:
        return self.tone == other.tone

    @abstractmethod
    def respell(self, stepAlt: int) -> Self:
        """Constructs an enharmonic of the current pitch with the given step alteration."""
        raise NotImplementedError

    def __abs__(self) -> Self:
        if self.step > 0 or self.step == 0 and self.acci >= 0:
            return self
        else:
            return -self

    def __lt__(self, other: Self) -> bool:
        return self.tone < other.tone or (
            self.tone == other.tone and self.step < other.step
        )

    def __eq__(self, other: Self):
        if not isinstance(other, self.__class__):
            return False
        return self.step == other.step and self.acci == other.acci

    @cachedGetter
    def __hash__(self):
        return hash((self.step, self.acci))

    def _m21_pitch(
        self,
        *,
        keepNatural: bool = False,
        useQuartertone: bool = True,
        roundToQuartertone: bool = True,
        round: bool = True,
        rmode: RMode | str = RMode.E,
    ) -> m21.pitch.Pitch:
        import music21 as m21

        m21_step = STEP_NAMES[self.step % 7]
        if self.acci.is_integer() or useQuartertone and self.acci % 0.5 == 0:
            m21_acci = self.acci
            m21_microtone = 0
        else:
            d = 0.5 if useQuartertone and roundToQuartertone else 1
            q, r = rdivmod(self.acci, d, round=round, rmode=rmode)
            m21_acci, m21_microtone = q * d, r * 100
        if not keepNatural and m21_acci == 0:
            m21_acci = None

        return m21.pitch.Pitch(
            step=m21_step,
            accidental=m21_acci,
            microtone=m21_microtone,
            octave=self.o + 4 if isinstance(self, _abc_base.Pitch) else None,
        )

    def _m21_interval(
        self,
        *,
        round: bool = True,
        rmode: RMode | str = RMode.D,
    ) -> m21.interval.Interval:
        import music21 as m21

        if isinstance(self, _abc_base.Pitch) and self.sgn < 0:
            p = -self
            neg = True
            if rmode == RMode.D or rmode == RMode.C:
                rmode = RMode.F
            elif rmode == RMode.U or rmode == RMode.F:
                rmode = RMode.C
        else:
            p = self
            neg = False
            if rmode == RMode.D:
                rmode = RMode.F
            elif rmode == RMode.U:
                rmode = RMode.C

        tone_int = rdiv(p.tone, round=round, rmode=rmode)
        acci_int = tone_int - cycGet(MAJOR_SCALE_TONES, p.step, 12)
        qual_int: int = clamp(-5, _qualMap(p.step, acci_int), 5)
        m21_specifier = _qual2Str_int(qual_int, augDimThresh=4)
        m21_diatonic = m21.interval.DiatonicInterval(
            m21_specifier, -p.step - 1 if neg else p.step + 1
        )
        m21_chromatic = m21.interval.ChromaticInterval(self.tone)
        return m21.interval.Interval(diatonic=m21_diatonic, chromatic=m21_chromatic)

    def m21(
        self,
        *,
        asInterval: bool = False,
        **kwargs,
    ) -> m21.pitch.Pitch | m21.interval.Interval:
        """
        Convert to a `music21.pitch.Pitch` or `music21.interval.Interval` object.

        **Note**: `music21` must be installed first for this method to work.
        """
        if importlib.util.find_spec("music21") is None:
            raise ImportError("`music21` must be installed first for the conversion.")
        if asInterval:
            return self._m21_interval(**kwargs)
        else:
            return self._m21_pitch(**kwargs)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo) -> Self:
        return self


class OPitch(Notation, _abc_edo.OPitch["Pitch"]):
    """
    Represents a general pitch, or equivalently, a simple interval, without octave
    specification.
    """

    __slots__ = ("_step", "_acci", "_hash")
    _step: int
    _acci: Real

    @cachedClassProp(key="_zero")
    def ZERO(cls) -> OPitch:
        """
        The C pitch, with no accidental, which is the identity element of addition in the
        octave pitch abelian group.
        """
        return cls._newHelper(0, 0)

    if t.TYPE_CHECKING:  # pragma: no cover

        @overload
        def __new__(cls, src: str) -> Self:
            """
            Creates a new `OPitch` object from string notation, in the form of either
            a pitch notation (e.g. `"C"`, `"F+"`, `"B-"`) or an interval notation (e.g. `"M3"`,
            `"m6"`, `"P5"`, `"A2"`).

            A formal definition of pitch notation and interval notation is displayed in the
            [extended BNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form)
            syntax below:

            ```ebnf
            accepted-input    = opitch-notation | interval-notation;

            opitch-notation   = note-name, acci;
            note-name         = letter-name | solfege-name | scale-degree;
            letter-name       = "C" | "D" | "E" | "F" | "G" | "A" | "B" |
                                ? any case variants of these items ?;
            solfege-name      = "do" | "ut" | "re" | "mi" | "fa" | "sol" | "la" | "si" | "ti" |
                                ? any case variants of these items ?;
            scale-degree      = "1" | "2" | "3" | "4" | "5" | "6" | "7";
            acci              = "" | symbolic-acci | numeric-acci;
            non-zero-digit    = "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9";
            digit             = "0" | non-zero-digit;
            positive-integer  = non-zero-digit, {digit};
            natural-number    = "0" | positive-integer;
            positive-decimal  = ([integer], ".", digit, {digit}) | (integer, ".");
            fraction          = natural-number, "/", positive-integer;
            sign              = "+" | "-";
            symbolic-acci     = {sign}; (* can be empty *)
            numeric-acci      = "[", sign, natural-number | fracion | positive-decimal, "]";

            interval-notation = [sign], quality, natural-number;
            quality           = quality-base, [numeric-acci];
            quality-base      = ("A", {"A"}) | "M" | "P" | "m" | ("d", {"d"}) | multiple-aug-dim;
            multiple-aug-dim  = ("[", "A" | "d", "*", natural-number, "]");
            ```
            """
            ...

        @overload
        def __new__(cls, m21_obj: m21.pitch.Pitch | m21.interval.Interval) -> Self:
            """
            Convert a `music21` pitch or interval object to an `OPitch` object.
            """
            ...

        @overload
        def __new__(cls, step: Integral | str, acci: Real | str = 0) -> Self:
            """
            Creates a new `OPitch` object from step and accidental values.
            """
            ...

        @overload
        def __new__(cls, step: Integral | str, *, tone: Real | Notation) -> Self:
            """
            Creates a new `OPitch` object from a step and a chromatic tone.

            If `tone` is a `PitchBase` object, this constructor is equivalent to
            `tone.opitch.respell(ostep(step) - tone.step)`, which constructs an enharmonic of
            `tone` with the given step.
            """
            ...

        @overload
        def __new__(
            cls,
            *,
            tone: Real | Notation,
            acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP,
        ) -> Self:
            """
            Creates a new `OPitch` object from a chromatic tone with accidental automatically
            chosen by accidental preference.
            """
            ...

    def __new__(
        cls,
        arg1=_MISSING,
        arg2=_MISSING,
        /,
        *,
        tone=_MISSING,
        acciPref=AcciPrefs.CLOSEST_FLAT_F_SHARP,
    ) -> Self:
        if arg2 is _MISSING:
            if arg1 is _MISSING:
                if tone is _MISSING:
                    raise TypeError(
                        "At least one positional argument or `tone` is required."
                    )
                # `tone` and `acciPref`
                return cls._fromTone(tone, acciPref)
            if tone is not _MISSING:
                # `step` and `tone`
                step = ostep(arg1)
                tone = _resolveTone(tone)
                return cls._fromStepAndTone(step, tone)
            if isinstance(arg1, str):
                # string notation
                return cls._parse(arg1)
            if isinstance(arg1, Notation):
                # another `PitchBase` object
                return arg1.opitch

            from .abc import wrapper as _abc_wrapper  # avoid cyclic import

            if isinstance(arg1, _abc_wrapper.Notation):
                return arg1.opitch._p
            if isinstance(arg1, _abc_base.DiatonicPitchBase):
                return cls._newHelper(arg1.opitch.step, arg1.acci)
            if lazyIsInstance(arg1, "music21.pitch.Pitch"):
                # `music21` pitch
                return cls._fromM21Pitch(arg1)
            if lazyIsInstance(arg1, "music21.interval.Interval"):
                # `music21` interval
                return cls._fromM21Interval(arg1)
            # `step` only
            step = ostep(arg1)
            return cls._newHelper(step, 0)

        # `step` and `acci`
        step = ostep(arg1)
        acci = _resolveAcci(arg2)
        return cls._newHelper(step, acci)

    @classmethod
    @lru_cache
    def _newHelper(cls, step: int, acci: Real) -> Self:
        # create a new instance with caching
        return cls._newImpl(step, acci)

    @classmethod
    def _newImpl(cls, step: int, acci: Real) -> Self:
        # the implementation of creating a new instance without caching
        self = super().__new__(cls)
        self._step = step
        self._acci = acci
        return self

    _fromStepAndAcci = _newHelper

    @classmethod
    @lru_cache
    def _fromStepAndTone(cls, step: int, tone: Real) -> Self:
        octave, step = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[step] - octave * 12
        return cls._newHelper(step, acci)

    @classmethod
    @lru_cache
    def _fromTone(
        cls,
        tone: Real,
        acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP,
    ) -> Self:
        step = acciPref(tone)
        octaves, ostep = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[ostep] - octaves * 12
        return cls._newHelper(step, acci)

    @classmethod
    def _parsePitch(cls, src: str) -> Self:
        return cls._newHelper(*_parseOPitch(src))

    @classmethod
    def _parseInterval(cls, src: str) -> Self:
        step, acci, neg = _parseInterval(src)
        step %= 7
        res = cls._newHelper(step, acci)
        if neg:
            res = -res
        return res

    @classmethod
    @lru_cache
    def _fromM21Pitch(cls, m21_obj: m21.pitch.Pitch):
        step = (ord(m21_obj.step) - 67) % 7  # 67 is the ASCII code of 'C'
        acci: Real = (
            0 if m21_obj.accidental is None else resolveInt(m21_obj.accidental.alter)
        )
        if not acci.is_integer():
            acci = Q(acci)  # quartertonal accidental
        microtoneCents = resolveInt(m21_obj.microtone.cents)
        if microtoneCents != 0:
            if microtoneCents.is_integer():
                acci += Q(microtoneCents, 100)
            else:
                acci += float(m21_obj.microtone.alter)
        return cls._newHelper(step, acci)

    @classmethod
    @lru_cache
    def _fromM21Interval(cls, m21_obj: m21.interval.Interval) -> Self:
        neg = m21_obj.direction < 0
        step = abs(m21_obj.generic.staffDistance) % 7
        cents = abs(m21_obj.cents)
        if cents.is_integer():
            tone = resolveInt(Q(int(cents), 100))
        else:
            tone = cents / 100
        result = cls._fromStepAndTone(step, tone)
        if neg:
            result = -result
        return result

    @property
    def step(self) -> int:
        return self._step

    @property
    def acci(self) -> Real:
        return self._acci

    @property
    @lru_cache
    def tone(self) -> Real:
        """
        Chromatic tone of the pitch, in half-steps.
        """
        return int(MAJOR_SCALE_TONES[self.step]) + self.acci

    def atOctave(self, octave: int = 0) -> Pitch:
        return Pitch._newHelper(self, octave - self.tone // 12)

    def withAcci(self, acci: Real = 0) -> Self:
        return self.__class__(self.step, acci)

    def isEnharmonic(self, other: Self) -> bool:
        return self.otone == other.otone

    def respell(self, stepAlt: int) -> Self:
        if stepAlt == 0:
            return self
        step = self.step + stepAlt
        octave, step = divmod(step, 7)
        acci = self.tone - int(MAJOR_SCALE_TONES[step]) - octave * 12
        return self._newHelper(step, acci)

    def __add__(self, other: Any) -> Self:
        if not isinstance(other, Notation):
            return NotImplemented
        step = self.step + other.step
        octave, step = divmod(step, 7)
        tone = self.tone + other.tone
        acci = tone - int(MAJOR_SCALE_TONES[step]) - octave * 12
        return self._newHelper(step, acci)

    def __neg__(self) -> Self:
        if self.step == 0:
            return self._newHelper(0, -self.acci)
        step = 7 - self.step
        tone = 12 - self.tone
        acci = tone - int(MAJOR_SCALE_TONES[step])
        return self._newHelper(step, acci)

    def __mul__(self, other: Any) -> Self:
        if not isinstance(other, Integral):
            return NotImplemented
        if other == 0:
            return self.ZERO
        if other == 1:
            return self
        step = self.step * other
        tone = self.tone * other
        octave, step = divmod(step, 7)
        acci = tone - int(MAJOR_SCALE_TONES[step]) - octave * 12
        return self._newHelper(step, acci)

    def __abs__(self) -> Self:
        return self

    # def __format__(self, format_spec: str) -> str:
    #     base, options = format_spec.split(";", 1)
    #     options = options.split(",")
    #     print(base, options)
    #     return str(self)

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[Any, ...]]:
        return (self._newImpl, (self.step, self.acci))


class Pitch(Notation, _abc_edo.Pitch[OPitch]):
    """
    Represents a pitch with specific octave, or an interval that may cross multiple octaves.
    """

    __slots__ = ("_opitch", "_o", "_hash")
    _opitch: OPitch
    _o: int

    @cachedClassProp(key="_zero")
    def ZERO(cls) -> Self:
        """
        The middle C pitch. It is the identity value of the pitch abelian group.
        """
        return cls._newHelper(OPitch.ZERO, 0)

    if t.TYPE_CHECKING:  # pragma: no cover
        # constructors inherited from `OPitch` but has an additional keyword argument `o`

        @overload
        def __new__(
            cls,
            src: str | _abc_diatonic.Notation | m21.pitch.Pitch | m21.interval.Interval,
            /,
            *,
            o: Integral,
        ) -> Self:
            """
            Equivalent to `Pitch(OPitch(src), o=o)`.
            """
            ...

        @overload
        def __new__(
            cls, step: Integral | str, acci: Real | str = 0, /, *, o: Integral
        ) -> Self:
            """
            Equivalent to `Pitch(OPitch(step, acci), o=o)`.
            """
            ...

        @overload
        def __new__(
            cls, step: Integral | str, /, *, tone: Real | Notation, o: Integral
        ) -> Self:
            """
            Eauivalent to `Pitch(OPitch(step, tone=tone), o=o)`.
            """
            ...

        @overload
        def __new__(
            cls,
            /,
            *,
            tone: Real | Notation,
            acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP,
            o: int,
        ) -> Self:
            """
            Eauivalent to `Pitch(OPitch(tone=tone, acciPref=acciPref), o=o)`
            """
            ...

        # new constructors for `Pitch`

        @overload
        def __new__(cls, src: str) -> Self:
            """
            Creates a new `Pitch` object from string notation, in the form of either a pitch
            notation (e.g. `"C_0"`, `"F+_-1"`, `"B-_2"`) or an interval notation (e.g. `"M3"`,
            `"m10"`, `"-P5"`, `"A9"`).

            A formal definition of pitch notation and interval notation is displayed in the
            [extended BNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form)
            syntax below:

            ```ebnf
            accepted-input    = pitch-notation | interval-notation;
            pitch-notation    = opitch-notation, ["_", [sign], integer];
            ```

            Where the meaning of some symbols are explained in the `OPitch` documentation.
            """
            ...

        @overload
        def __new__(cls, opitch: OPitch, /, *, o: int = 0) -> Self:
            """
            Creates a `Pitch` object by putting an `OPitch` into a specific *nominal* octave.
            """
            ...

        @overload
        def __new__(cls, step: int | str, acci: Real | str = 0, /) -> Self:
            """
            Creates a `Pitch` object from a step and an accidental. The step value is octave
            sensitive.

            Equivalent to `Pitch(OPitch(step % 7, acci), o=step // 7)`.
            """
            ...

        @overload
        def __new__(cls, step: int | str, /, *, tone: Real | Notation) -> Self:
            """
            Creates a `Pitch` object from a step and a chromatic tone. Both the step and tone
            are octave sensitive.

            Denote `o = step // 7`, then this constructor is equivalent to
            `Pitch(OPitch(step % 7, tone=tone - o * 12), o=o)`.
            """
            ...

        @overload
        def __new__(
            cls,
            /,
            *,
            tone: Real | Notation,
            acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP,
        ) -> Self:
            """
            Creates a `Pitch` object from a chromatic tone with accidental automatically
            chosen by accidental preference. The tone value is octave sensitive.
            """
            ...

    def __new__(
        cls,
        arg1=_MISSING,
        arg2=_MISSING,
        /,
        *,
        o=_MISSING,
        tone=_MISSING,
        acciPref=_MISSING,
    ) -> Self:
        if o is not _MISSING:
            # call `OPitch` constructor with other arguments except `o`
            opitch = OPitch(arg1, arg2, tone=tone, acciPref=acciPref)
            o = int(o)
            return cls._newHelper(opitch, o)

        if arg2 is _MISSING:
            if arg1 is _MISSING:
                # `tone` and `acciPref`
                if acciPref is _MISSING:
                    acciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP
                tone = _resolveTone(tone)
                return cls._fromTone(tone, acciPref)
            if acciPref is not _MISSING:
                warnings.warn(
                    "`acciPref` is ignored when at least one positional argument is given."
                )
            if tone is not _MISSING:
                # `step` and `tone`
                step = _resolveStep(arg1)
                tone = _resolveTone(tone)
                return cls._fromStepAndTone(step, tone)
            if isinstance(arg1, str):
                # string notation
                return cls._parse(arg1)
            if isinstance(arg1, Pitch):
                # another `Pitch` object
                return arg1
            if isinstance(arg1, Notation):
                # `OPitch` object
                return cls._newHelper(arg1.opitch, 0)

            from .abc import wrapper as _abc_wrapper  # avoid cyclic import

            if isinstance(arg1, _abc_wrapper.Notation):
                return cls._newHelper(arg1.opitch._p, 0)
            if isinstance(arg1, _abc_diatonic.Notation):
                return cls._fromStepAndAcci(arg1.step, arg1.acci)
            if lazyIsInstance(arg1, "music21.pitch.Pitch"):
                # `music21` pitch
                return cls._fromM21Pitch(arg1)
            if lazyIsInstance(arg1, "music21.interval.Interval"):
                # `music21` interval
                return cls._fromM21Interval(arg1)

            # `step` only
            step = _resolveStep(arg1)
            return cls._fromStepAndAcci(step, 0)

        if tone is not _MISSING or acciPref is not _MISSING:
            warnings.warn(
                "`tone` and`acciPref` are ignored when `step` and `acci` are given."
            )

        # `step` and `acci`
        step = _resolveStep(arg1)
        acci = _resolveAcci(arg2)
        return cls._fromStepAndAcci(step, acci)

    @classmethod
    def _fromTone(cls, tone: Real, acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP):
        o, otone = divmod(tone, 12)
        opitch = OPitch._fromTone(otone, acciPref)
        return cls._newHelper(opitch, o)

    @classmethod
    @lru_cache
    def _fromStepAndAcci(cls, step: int, acci: Real) -> Self:
        o, ostep = divmod(step, 7)
        opitch = OPitch._newHelper(ostep, acci)
        return cls._newHelper(opitch, o)

    @classmethod
    @lru_cache
    def _fromStepAndTone(cls, step: int, tone: Real) -> Self:
        o, ostep = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[ostep] - o * 12
        opitch = OPitch._newHelper(ostep, acci)
        return cls._newHelper(opitch, o)

    @classmethod
    @lru_cache
    def _parsePitch(cls, src: str) -> Self:
        if not src[-1].isdigit():
            # octave not specified, assume octave 0
            opitch = OPitch._parse(src)
            o = 0
        else:
            opitch, o = src.rsplit("_", 1)
            opitch = OPitch._parsePitch(opitch)
            o = int(o)
        return Pitch._newHelper(opitch, o)

    @classmethod
    def _parseInterval(cls, src: str) -> Self:
        step, acci, neg = _parseInterval(src)
        result = cls._fromStepAndAcci(step, acci)
        if neg:
            result = -result
        return result

    @classmethod
    def _fromM21Pitch(cls, m21_obj: m21.pitch.Pitch) -> Self:
        opitch = OPitch._fromM21Pitch(m21_obj)
        o = m21_obj.implicitOctave - 4
        return cls._newHelper(opitch, o)

    @classmethod
    @lru_cache
    def _fromM21Interval(cls, m21_obj: m21.interval.Interval) -> Self:
        step = m21_obj.generic.staffDistance
        tone = m21_obj.chromatic.semitones
        if not tone.is_integer() and (cents := m21_obj.chromatic.cents).is_integer():
            tone = Q(int(cents), 100)
        return cls._fromStepAndTone(step, tone)

    @property
    def opitch(self) -> OPitch:
        return self._opitch

    @property
    def otone(self) -> Real:
        return self.opitch.otone

    @property
    def o(self) -> int:
        return self._o

    @property
    def acci(self) -> Real:
        return self.opitch.acci

    @property
    @lru_cache
    def tone(self) -> Real:
        """
        Chromatic tone of the pitch, in half-steps. Equals to the chromatic tone of octave
        pitch plus nominal octave times 12.
        """
        return self.opitch.tone + self.o * 12

    def withAcci(self, acci: Real = 0) -> Self:
        return self._newHelper(self.opitch.withAcci(acci), self.o)

    def respell(self, stepAlt: int) -> Self:
        if stepAlt == 0:
            return self
        step = self.step + stepAlt
        octave, ostep = divmod(step, 7)
        acci = self.tone - MAJOR_SCALE_TONES[ostep] - octave * 12
        return self._newHelper(OPitch._newHelper(ostep, acci), octave)

    def __add__(self, other: Any) -> Self:
        if not isinstance(other, Notation):
            return NotImplemented
        step = self.step + other.step
        tone = self.tone + other.tone
        octave, ostep = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[ostep] - octave * 12
        return self._newHelper(OPitch._newHelper(ostep, acci), octave)

    def __neg__(self) -> Self:
        if self.opitch.step == 0:
            return self._newHelper(-self.opitch, -self.o)
        return self._newHelper(-self.opitch, -self.o - 1)

    def __mul__(self, other: int) -> Self:
        if not isinstance(other, Integral):
            return NotImplemented
        if other == 0:
            return self.C_0
        if other == 1:
            return self
        step = self.step * other
        tone = self.tone * other
        o, ostep = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[ostep] - o * 12
        return self._newHelper(OPitch._newHelper(ostep, acci), o)

    @cachedGetter
    def __hash__(self) -> int:
        return hash((self.step, self.acci))

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[Any, ...]]:
        return (self._newImpl, (self.opitch, self.o))


oP = OPitch
P = Pitch
