from __future__ import annotations

import itertools as it
import re
import typing as t
from abc import ABCMeta, abstractmethod
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator, Sequence, Set
from functools import lru_cache
from numbers import Integral, Real
from typing import Self, overload, Protocol, Any, Never, Literal
from fractions import Fraction as Q

import numpy as np
from sortedcontainers import SortedSet
import pyrsistent as pyr
from bidict import bidict

if t.TYPE_CHECKING:
    import music21 as m21  # pragma: no cover

from .utils.cls import classconst
from .utils.number import rdiv, rdivmod, rbisect, RMode, resolveInt

# TODO)) put utils in a separate library

__all__ = [
    "ostep",
    "step",
    "acci",
    "AcciPref",
    "DegMaps",
    "AcciPrefs",
    "Modes",
    "PitchBase",
    "OPitch",
    "DegMap",
    "PitchBase",
    "OPitch",
    "Pitch",
    "Mode",
    "Scale",
    "STEPS_CO5",
    "MAJOR_SCALE_TONES_CO5",
    "MAJOR_SCALE_TONES",
    "MAJOR_SCALE_TONES_SET",
    "STEP_NAMES",
    "STEP_NAMES_SET",
]

_MISSING = object()


STEPS_CO5: Sequence[int] = np.arange(-1, 6) * 4 % 7
"""
major scale steps in circle of fifths order

*Value*: `np.array([3, 0, 4, 1, 5, 2, 6])`
"""
STEPS_CO5.flags.writeable = False

MAJOR_SCALE_TONES_CO5: Sequence[int] = np.arange(-1, 6) * 7 % 12
"""major scale tones in circle of fifths order"""
MAJOR_SCALE_TONES_CO5.flags.writeable = False

MAJOR_SCALE_TONES: Sequence[int] = np.sort(MAJOR_SCALE_TONES_CO5)
"""major scale tones in increasing order"""
MAJOR_SCALE_TONES.flags.writeable = False

MAJOR_SCALE_TONES_SET = frozenset(MAJOR_SCALE_TONES)
"""major scale tones as a set"""

PERFECT_INTERVAL_STEPS = frozenset((0, 3, 4))
STEP_NAMES: Sequence[str] = np.roll(np.array([chr(65 + i) for i in range(7)]), -2)
"""step names from C to B"""
STEP_NAMES.flags.writeable = False
STEP_NAMES_SET = frozenset(STEP_NAMES)
"""step names as a set"""
STEP_NAMES_CO5: Sequence[str] = STEP_NAMES[STEPS_CO5]
"""step names in circle of fifths order"""
STEP_NAMES_CO5.flags.writeable = False
SOLFEGE_NAMES: Sequence[str] = np.array(["do", "re", "mi", "fa", "sol", "la", "si"])
"""solfège names"""
SOLFEGE_NAMES.flags.writeable = False
SOLFEGE_NAMES_SET = frozenset(SOLFEGE_NAMES)
"""solfège names as a set"""
SOLFEGE_NAMES_CO5: Sequence[str] = SOLFEGE_NAMES[STEPS_CO5]
"""solfège names in circle of fifths order"""
SOLFEGE_NAMES_CO5.flags.writeable = False
_sofegeNamesInvMap = pyr.pmap(
    {name: i for i, name in enumerate(SOLFEGE_NAMES)} | {"ut": 0, "ti": 6}
)
_intervalQualityMap = bidict(
    (
        ("d", -2),  # diminished
        ("m", -1),  # minor
        ("P", 0),  # perfect
        ("M", 1),  # major
        ("A", 2),  # augmented
    )
)

_leadingAlphaRe = re.compile(r"^(?:[a-z]+|[1-7])", re.IGNORECASE)
_trailingDigitsRe = re.compile(r"\d+$")
_multipleAugRe = re.compile(r"^A+")
_multipleAugNumRe = re.compile(r"^\[A\*(\d+)\]")
_multipleDimRe = re.compile(r"^d+")
_multipleDimNumRe = re.compile(r"^\[d\*(\d+)\]")

type AcciPref = Callable[[Real], int]
"""
Type alias for a function that takes a tone in half-steps and returns the preferred step for 
that tone. The accidental can be later computed by taking the difference between the given
tone and the standard reference tone in C major scale. Some predefined accidental preference
rules can be found in `AcciPrefs`.
"""


def _parseMicrotone(src: str) -> Real:
    if len(src) == 0:
        return 0
    valueSrc = src[1:-1]  # extract content inside the brackets
    if "/" in valueSrc:  # fractional value
        return resolveInt(Q(valueSrc))
    else:  # float value
        return resolveInt(float(valueSrc))


def _parseAcci(src: str) -> int:
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
        step = _sofegeNamesInvMap.get(src)
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
        octave = int(octave)
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
    step = int(src[end:]) - 1
    qual = _parseQual(src[:end], step)
    acci = _qualInvMap(step, qual)
    return step, acci, neg


def _acci2Str(acci: Real) -> str:
    if acci == 0:
        return ""
    elif acci > 0:
        if acci.is_integer():
            acci = int(acci)
            if acci <= 3:
                return "+" * acci
            else:
                return f"[{acci:+d}]"
        else:
            return f"[{round(acci, 3):+}]"
    else:
        if acci.is_integer():
            acci = int(acci)
            if acci >= -3:
                return "-" * abs(acci)
            else:
                return f"[{acci:+d}]"
        else:
            return f"[{round(acci, 3):+}]"


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


def _prefectQualMap(acci: Real) -> Real:
    """Mapping from accidental to interval quality for "perfect" intervals."""
    if acci >= 1:
        return acci + 1
    elif acci >= -1:
        return 2 * acci
    else:
        return acci - 1


def _majorQualMap(acci: Real) -> Real:
    """Mapping from accidental to interval quality for "major" intervals."""
    if acci >= 0:
        return acci + 1
    elif acci >= -1:
        return 2 * acci + 1
    else:
        return acci


def _qualMap(step: int, acci: Real) -> Real:
    if step in PERFECT_INTERVAL_STEPS:
        return _prefectQualMap(acci)
    else:
        return _majorQualMap(acci)


def _prefectQualInvMap(qual: Real) -> Real:
    """Mapping from interval quality to accidental for "perfect" intervals."""
    if qual >= 2:
        return qual - 1
    elif qual >= -2:
        return resolveInt(qual / 2)
    else:
        return qual + 1


def _majorQualInvMap(qual: Real) -> Real:
    """Mapping from interval quality to accidental for "major" intervals."""
    if qual >= 1:
        return qual - 1
    elif qual >= -1:
        return resolveInt((qual - 1) / 2)
    else:
        return qual


def _qualInvMap(step: int, qual: Real) -> Real:
    if step in PERFECT_INTERVAL_STEPS:
        return _prefectQualInvMap(qual)
    else:
        return _majorQualInvMap(qual)


def _qual2Str_int(qual: int) -> str:
    """Interval quality to string, but for integers only."""
    if qual > 4:
        return f"[A*{qual - 1}]"
    elif qual > 1:
        return "A" * (qual - 1)
    elif qual < -4:
        return f"[d*{-qual - 1}]"
    elif qual < -1:
        return "d" * (-qual - 1)
    else:
        return _intervalQualityMap.inv[qual]


def _qual2Str(qual: Real, /, rmode: RMode | str = RMode.D) -> str:
    """Interval quality to string."""
    q, r = rdivmod(qual, 1, rmode=rmode)
    if r == 0:
        return _qual2Str_int(q)
    else:
        return f"{_qual2Str_int(q)}[{round(r, 3):+}]"


def ostep(src: int | str) -> int:
    """Resolves an octave step representation to an integer between 0 and 6."""
    if isinstance(src, str):
        return _parseOStep(src)
    else:
        return src % 7


def step(src: int | str) -> int:
    """Resolves a step representation in specific octave to an integer."""
    if isinstance(src, str):
        return _parseStep(src)
    else:
        return src


def acci(src: Real | str) -> Real:
    """Resolves an accidental representation to a semitone value."""
    if isinstance(src, str):
        return _parseAcci(src)
    else:
        return resolveInt(src)


# aliases to avoid name conflict
_resolveStep = step
_resolveAcci = acci


def _resolveTone(src: PitchBase | Real) -> Real:
    if isinstance(src, PitchBase):
        return src.tone
    else:
        return resolveInt(src)


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

    def __init__(self) -> Never:
        raise TypeError("This class is not intended to be instantiated")


class SupportsGetItem(Protocol):
    def __getitem__(self, key: int) -> Self: ...


class DegMap(Sequence[Real]):
    """A mapping from scale degrees to tones in an octave."""

    __slots__ = ("_tones", "_pitches", "_mode")

    if t.TYPE_CHECKING:

        @overload
        def __new__(cls, tones: Iterable[Real]): ...

        @overload
        def __new__(cls, *tones: Real): ...

        @overload
        def alter(self, idx: int, acci: Real) -> Self: ...

        @overload
        def alter(self, acci: Iterable[Real]) -> Self: ...

        @overload
        def alter(self, idx: Iterable[int], acci: Real | Iterable[Real]) -> Self: ...

    def __new__(cls, *args) -> Self:
        if len(args) == 1 and isinstance(args[0], Iterable):
            tones = np.array(tuple(args[0]))
        else:
            tones = np.array(args)
        if len(tones) != 7:
            raise ValueError(f"Expected 7 tones, got {len(tones)}")

        # The first element is always 0. Omitting it can save space, but will bring
        # more trouble working with the array later on.
        # So here I choose not to omit it.

        tones -= tones[0]
        return cls._newHelper(tones)

    @classmethod
    def _newHelper(cls, tones: np.ndarray):
        # caching is not enabled because `numpy` array is not hashable
        self = super().__new__(cls)
        tones.flags.writeable = False
        self._tones = tones
        return self

    @property
    def p(self) -> Sequence[OPitch]:
        """Access degree map elements as pitch objects."""
        if not hasattr(self, "_pitches"):
            self._pitches = _DegMapPitchesView(self)
        return self._pitches

    @property
    def mode(self) -> Mode:
        """Turn the degree map into a mode."""
        if not hasattr(self, "_mode"):
            self._mode = Mode(self.p)
        return self._mode

    def __len__(self):
        return len(self._tones)

    def __getitem__(
        self, key: int | str | OPitch | slice | Iterable[int]
    ) -> int | Sequence[int]:
        if isinstance(key, tuple):
            key = np.array(key)
        if isinstance(key, Integral):
            return self._tones[key]
        else:
            key = OPitch(key)
            return self._tones[key.deg] + key.acci

    def __iter__(self):
        return iter(self._tones)

    def __reversed__(self):
        return reversed(self._tones)

    def __hash__(self):
        return hash(self._tones.tobytes())

    def __eq__(self, other: Any):
        if not isinstance(other, DegMap):
            return False
        return np.all(self._tones == other._tones)

    def __add__(self, other: Iterable[Real]) -> Self:
        if not isinstance(other, np.ndarray):
            if not isinstance(other, Sequence):
                other = tuple(other)
            other = np.array(other)
        newTones = self._tones + other
        if newTones[0] != 0:
            newTones -= newTones[0]
        return self._newHelper(newTones)

    def alter(
        self,
        arg1: int | Iterable[int] | Iterable[Real],
        arg2: Real | Iterable[Real] | None = None,
    ) -> Self:
        if arg2 is None:
            if isinstance(arg1, Iterable):
                return self + arg1
            else:
                return self
        if not isinstance(arg1, Iterable):
            arg1 = (arg1,)
        if not isinstance(arg2, Iterable):
            arg2 = it.repeat(arg2)
        # store alteration amounts in a list to avoid `numpy` array type issues
        lst = [0 for _ in range(7)]
        for idx, acci in zip(arg1, arg2):
            lst[idx % 7] += acci
        return self + lst

    def roll(self, shift: int) -> Self:
        newTones = np.roll(self._tones, shift)
        newTones -= newTones[0]
        newTones %= 12
        return self._newHelper(newTones)

    def diff(self) -> np.ndarray[int]:
        return np.diff(self._tones, append=12)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._tones})"


class DegMaps:
    MAJOR = IONIAN = DegMap(MAJOR_SCALE_TONES)
    HARMONIC_MAJOR = MAJOR.alter(5, -1)
    DORIAN = MAJOR.roll(-1)
    PHRYGIAN = MAJOR.roll(-2)
    LYDIAN = MAJOR.roll(1)
    MIXOLYDIAN = MAJOR.roll(2)
    MINOR = AEOLIAN = MAJOR.roll(3)
    HARMONIC_MINOR = MINOR.alter(6, 1)
    MELODIC_MINOR = HARMONIC_MINOR.alter(5, 1)
    LOCRIAN = MAJOR.roll(4)


class _DegMapPitchesView(Sequence["OPitch"], Set["OPitch"]):
    __slots__ = ("_parent",)

    def __init__(self, parent: DegMap):
        self._parent = parent

    def __len__(self):
        return 7

    def __iter__(self) -> Iterator["OPitch"]:
        return map(OPitch._fromStepAndTone, range(7), self._parent._tones)

    def __contains__(self, value: Any):
        if not isinstance(value, PitchBase):
            return False
        return self._parent._tones[value.step % 7] == value.tone

    def index(self, value: PitchBase) -> OPitch:
        """Find the degree of pitch in the degree map."""
        value = value.opitch
        step = value.step
        acci = value.tone - self._parent._tones[step]
        return OPitch._newHelper(step, acci)

    def __eq__(self, other) -> bool:
        return self._parent == other._parent

    def __getitem__(self, key: int) -> int:
        if isinstance(key, slice):
            return self._slice(key)
        if isinstance(key, Iterable) and not isinstance(key, str):
            return self._multiIndex(key)
        return self._getItem(key)

    def _getItem(self, key: int | str | OPitch) -> OPitch:
        if isinstance(key, Integral):
            return OPitch._fromStepAndTone(key, self._parent._tones[key])
        if not isinstance(key, OPitch):
            key = OPitch(key)
        return OPitch._fromStepAndTone(
            key.step, self._parent._tones[key.step] + key.acci
        )

    def _slice(self, key: slice) -> Sequence[OPitch]:
        steps = np.arange(7)[key]
        tones = self._parent._tones[steps]
        return np.array(
            [OPitch._fromStepAndTone(s, t) for s, t in zip(steps, tones)], dtype=object
        )

    def _multiIndex(self, key: Iterable[int | str | OPitch]) -> Sequence[OPitch]:
        return np.array([self._getItem(k) for k in key], dtype=object)


class PitchBase(metaclass=ABCMeta):
    """Base class for `OPitch` and `Pitch`."""

    __slots__ = ()

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
            try:  # assume to be an interval notation
                return cls._parseInterval(src)
            except ValueError:
                raise ValueError(f"invalid pitch or interval string: {src}")

    @property
    def step(self) -> int:
        """
        Degree of the pitch in C major scale, not considering accidentals. Equals to the step
        of octave pitch plus nominal octave times 7.
        """
        return self.opitch.step + self.o * 7

    @property
    @abstractmethod
    def acci(self) -> Real:
        """An accidental value in semitones that modifies the pitch."""
        raise NotImplementedError

    @property
    def tone(self) -> Real:
        print("PitchBase.tone() called")
        octave, ostep = divmod(self.step, 7)
        return MAJOR_SCALE_TONES[ostep] + self.acci + octave * 12

    @property
    @abstractmethod
    def opitch(self) -> OPitch:
        raise NotImplementedError

    @property
    def otone(self) -> Real:
        return self.tone % 12

    @property
    def o(self) -> int:
        """
        Returns the nominal octave of the pitch, determined solely by step.

        e.g. `Pitch("B+_0").o` and `Pitch("C-_0").o` are both `0`.
        """
        return int(self.step // 7)

    @property
    def octave(self) -> int:
        """
        Returns the actual octave of the pitch, determined by the tone instead of step.

        e.g. `Pitch("B+_0").octave` is `1` and `Pitch("C-_0").octave` is `-1`.
        """
        return int(self.tone // 12)

    @property
    def qual(self) -> Real:
        """
        Interval quality when regarding the pitch as an interval.
        """
        return self.opitch.qual

    @property
    def freq(self) -> float:
        """
        Frequency of the pitch relative to middle C.
        """
        return np.pow(2, self.tone / 12)

    def interval(self, *, rmode: RMode | str = RMode.D, **kwargs) -> str:
        "String representation of the pitch as an interval."
        return f"{_qual2Str(self.qual, rmode=rmode)}{self.step + 1}"

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
        return self.opitch.atOctave(octave)

    def isEnharmonic(self, other: Self) -> bool:
        return self.tone == other.tone

    @abstractmethod
    def respell(self, stepAlt: int) -> Self:
        """Constructs an enharmonic of the current pitch with the given step alteration."""
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        raise NotImplementedError

    def __abs__(self) -> Self:
        if self.step > 0 or self.step == 0 and self.acci >= 0:
            return self
        else:
            return -self

    @abstractmethod
    def __neg__(self) -> Self:
        raise NotImplementedError

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __lt__(self, other: Self) -> bool:
        return self.tone < other.tone or (
            self.tone == other.tone and self.step < other.step
        )

    def __eq__(self, other: Self):
        if not isinstance(other, self.__class__):
            return False
        return self.step == other.step and self.acci == other.acci

    def m21(
        self,
        *,
        keepNatural: bool = False,
        useQuartertone: bool = True,
        roundToQuartertone: bool = True,
        round: bool = True,
        rmode: RMode | str = RMode.E,
        asInterval: bool = False,
    ) -> m21.pitch.Pitch:
        """
        Convert to a `music21.pitch.Pitch` or `music21.interval.Interval` object.

        **Note**: `music21` must be installed first for this method to work.
        """
        try:
            import music21 as m21
        except ImportError:
            raise ImportError("`music21` must be installed first for the conversion.")
        print(self.step)
        if asInterval:
            qual = rdiv(self.qual, rmode=rmode, round=round)
            m21_diatonic = m21.interval.DiatonicInterval(
                _qual2Str_int(qual), self.step + 1
            )
            m21_chromatic = m21.interval.ChromaticInterval(self.tone)
            return m21.interval.Interval(diatonic=m21_diatonic, chromatic=m21_chromatic)
        else:
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
                octave=self.o + 4 if isinstance(self, Pitch) else None,
            )

        # TODO)) Support for asInterval


class OPitch(PitchBase):
    """
    Represents a pitch in an octave, or equivalently, an interval no greater than an octave,
    which is often referred to as a **pitch / interval class**.
    """

    __slots__ = ("_step", "_acci")

    if t.TYPE_CHECKING:

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
        def __new__(cls, step: int | str, acci: Real | str = 0) -> Self:
            """
            Creates a new `OPitch` object from step and accidental values.
            """
            ...

        @overload
        def __new__(cls, deg: PitchBase, acci: Real | str = 0) -> Self:
            """
            Creates a new `OPitch` object from degree and accidental. If the degree object
            already has an accidental, it will be added to the given `acci` value.

            If `deg` is of type `PitchBase`, this constructor is equivalent to
            `deg.opitch.alter(acci)`.
            """
            ...

        @overload
        def __new__(cls, step: int | str, *, tone: Real | PitchBase) -> Self:
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
            tone: Real | PitchBase,
            acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP,
        ) -> Self:
            """
            Creates a new `OPitch` object from a chromatic tone with accidental automatically
            chosen by accidental preference.
            """
            ...

    @classconst
    def C(cls) -> Self:
        """
        The C pitch, with no accidental, which is the identity element of addition in the
        octave pitch abelian group.
        """
        return cls._newHelper(0, 0)

    @classmethod
    def co5(cls, n: int = 0) -> Self:
        """
        Returns the `n`-th pitch in circle of fifths order, starting from C.
        positive `n` values means `n` perfect fifths up while negative `n` values means `n`
        perfect fifths down. This method is equivalent to `OPitch("P5") * n`.

        When `n` ranges from -7 to 7, this method yields the tonic of major scale with `abs(n)`
        sharps (for positive `n`) or flats (for negative `n`).
        """
        step = n * 4 % 7
        acci = (n + 1) // 7
        return cls._newHelper(step, acci)

    def __new__(
        cls,
        arg1=_MISSING,
        arg2=_MISSING,
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
            if isinstance(arg1, PitchBase):
                # another `PitchBase` object
                return arg1.opitch
            # `step` only
            step = ostep(arg1)
            return cls._newHelper(step, 0)

        acci = _resolveAcci(arg2)
        if isinstance(arg1, PitchBase):
            # `PitchBase` and `acci`
            return arg1.opitch.alter(acci)

        # `step` and `acci`
        step = ostep(arg1)
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

    @property
    def step(self) -> int:
        return self._step

    @property
    def acci(self) -> Real:
        return self._acci

    @property
    def opitch(self) -> OPitch:
        return self

    @property
    def o(self) -> int:
        return 0

    @property
    @lru_cache
    def tone(self) -> Real:
        """
        Chromatic tone of the pitch, in half-steps.
        """
        print("OPitch.tone() called")
        return MAJOR_SCALE_TONES[self.step] + self.acci

    @property
    @lru_cache
    def qual(self) -> Real:
        return _qualMap(self.step, self.acci)

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
        acci = self.tone - MAJOR_SCALE_TONES[step] - octave * 12
        return self._newHelper(step, acci)

    def __add__(self, other: Self) -> Self:
        step = self.step + other.step
        octave, step = divmod(step, 7)
        tone = self.tone + other.tone
        acci = tone - MAJOR_SCALE_TONES[step] - octave * 12
        return self._newHelper(step, acci)

    def __neg__(self) -> Self:
        if self.step == 0:
            return self._newHelper(0, -self.acci)
        step = 7 - self.step
        tone = 12 - self.tone
        acci = tone - MAJOR_SCALE_TONES[step]
        return self._newHelper(step, acci)

    def __mul__(self, other: int) -> Self:
        if other == 0:
            return self.C
        if other == 1:
            return self
        if not isinstance(other, Integral):
            raise TypeError(f"can only multiply by integers, got {type(other)}")
        step = self.step * other
        tone = self.tone * other
        octave, step = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[step] - octave * 12
        return self._newHelper(step, acci)

    def __abs__(self) -> Self:
        return self

    def __str__(self) -> str:
        return f"{STEP_NAMES[self.step]}{_acci2Str(self.acci)}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    def __hash__(self) -> int:
        return hash((self.step, self.acci))

    def __reduce__(self):
        return (self._newImpl, (self.step, self.acci))


class Pitch(PitchBase):
    """
    Represents a pitch with specific octave, or an interval that may cross multiple octaves.
    """

    __slots__ = ("_opitch", "_o")

    if t.TYPE_CHECKING:
        # constructors inherited from `OPitch` but has an additional keyword argument `o`

        @overload
        def __new__(cls, src: str, *, o: int) -> Self:
            """
            Equivalent to `Pitch(OPitch(src), o=o)`.
            """
            ...

        @overload
        def __new__(cls, step: int | str, acci: Real | str = 0, *, o: int) -> Self:
            """
            Equivalent to `Pitch(OPitch(step, acci), o=o)`.
            """
            ...

        @overload
        def __new__(cls, deg: PitchBase, acci: Real | str = 0, *, o: int) -> Self:
            """
            Equivalent to `Pitch(OPitch(deg, acci), o=o)`.
            """
            ...

        @overload
        def __new__(cls, step: int | str, *, tone: Real | PitchBase, o: int) -> Self:
            """
            Eauivalent to `Pitch(OPitch(step, tone=tone), o=o)`.
            """
            ...

        @overload
        def __new__(
            cls,
            *,
            tone: Real | PitchBase,
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
        def __new__(cls, opitch: OPitch, *, o: int = 0) -> Self:
            """
            Creates a `Pitch` object by putting an `OPitch` into a specific *nominal* octave.
            """
            ...

        @overload
        def __new__(cls, step: int | str, acci: Real | str = 0) -> Self:
            """
            Creates a `Pitch` object from a step and an accidental. The step value is octave
            sensitive.

            Equivalent to `Pitch(OPitch(step % 7, acci), o=step // 7)`.
            """
            ...

        @overload
        def __new__(cls, deg: PitchBase, acci: Real | str = 0) -> Self:
            """
            Creates a `Pitch` object from a degree and an accidental. If the degree object
            already has an accidental, it will be added to the given `acci` value.
            """
            ...

        @overload
        def __new__(cls, step: int | str, *, tone: Real | PitchBase) -> Self:
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
            *,
            tone: Real | PitchBase,
            acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP,
        ) -> Self:
            """
            Creates a `Pitch` object from a chromatic tone with accidental automatically
            chosen by accidental preference. The tone value is octave sensitive.
            """
            ...

    @classconst
    def C_0(cls) -> Self:
        """
        The middle C pitch. It is the identity value of the pitch abelian group.
        """
        return cls._newHelper(OPitch.C, 0)

    def __new__(
        cls,
        arg1=_MISSING,
        arg2=_MISSING,
        *,
        o=_MISSING,
        tone=_MISSING,
        acciPref=AcciPrefs.CLOSEST_FLAT_F_SHARP,
    ) -> Self:
        if o is not _MISSING:
            # call `OPitch` constructor with other arguments except `o`
            opitch = OPitch(arg1, arg2, tone=tone, acciPref=acciPref)
            return cls._newHelper(opitch, o)

        if arg2 is _MISSING:
            if arg1 is _MISSING:
                # `tone` and `acciPref`
                tone = _resolveTone(tone)
                return cls._fromTone(tone, acciPref)
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
            if isinstance(arg1, PitchBase):
                # `PitchBase` object
                return cls._newHelper(arg1.opitch, 0)

            # `step` only
            step = _resolveStep(arg1)
            return cls._fromStepAndAcci(step, 0)

        acci = _resolveAcci(arg2)
        if isinstance(arg1, Pitch):
            # a `Pitch` object and `acci`
            return arg1.alter(acci)
        if isinstance(arg1, PitchBase):
            return cls._newHelper(arg1.opitch.alter(acci), 0)

        # `step` and `acci`
        step = _resolveStep(arg1)
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
            octave = 0
        else:
            opitch, octave = src.rsplit("_", 1)
            opitch = OPitch._parsePitch(opitch)
            octave = int(octave)
        return Pitch._newHelper(opitch, octave)

    @classmethod
    def _parseInterval(cls, src: str) -> Self:
        step, acci, neg = _parseInterval(src)
        result = cls._fromStepAndAcci(step, acci)
        if neg:
            result = -result
        return result

    @classmethod
    @lru_cache
    def _newHelper(cls, opitch: OPitch, o: int) -> Self:
        return cls._newImpl(opitch, o)

    @classmethod
    def _newImpl(cls, opitch: OPitch, o: int) -> Self:
        """The fundamental constructor of `Pitch` class."""
        self = super().__new__(cls)
        self._opitch = opitch
        self._o = o
        return self

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

    @property
    def qual(self) -> Real:
        return self.opitch.qual

    @property
    # @lru_cache
    def sgn(self) -> Literal[-1, 0, 1]:
        if self.step == 0:
            return (self.acci < 0) - (self.acci > 0)
        return (self.step < 0) - (self.step > 0)

    def interval(self, *, compound: bool = False, **kwargs):
        if self < self.C_0:  # negative interval
            return f"-{(-self).interval(compound=compound, **kwargs)}"
        else:
            return super().interval(compound=compound, **kwargs)

    def withAcci(self, acci: Real = 0) -> Self:
        return self._newHelper(self.opitch.withAcci(acci), self.o)

    def respell(self, stepAlt: int) -> Self:
        if stepAlt == 0:
            return self
        step = self.step + stepAlt
        octave, ostep = divmod(step, 7)
        acci = self.tone - MAJOR_SCALE_TONES[ostep] - octave * 12
        return self._newHelper(OPitch._newHelper(ostep, acci), octave)

    def hz(self, middleA: float = 440) -> float:
        """
        Returns the frequency of the pitch in hertz.
        """
        return middleA * np.power(2, (self.tone - 9) / 12)

    def __add__(self, other: PitchBase) -> Self:
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
        if other == 0:
            return self.C_0
        if other == 1:
            return self
        if not isinstance(other, Integral):
            raise TypeError(
                f"can only multiply by integers, got {other.__class__.__name__}"
            )
        step = self.step * other
        tone = self.tone * other
        octave, ostep = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[ostep] - octave * 12
        return self._newHelper(OPitch._newHelper(ostep, acci), octave)

    def __str__(self):
        return f"{STEP_NAMES[self.opitch.step]}{_acci2Str(self.acci)}_{self.o}"

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"

    @lru_cache
    def __hash__(self) -> int:
        return hash((self.opitch, self.o))


def _modeAlter(
    pitches: np.ndarray[OPitch], step: int, acci: Real
) -> np.ndarray[OPitch]:
    if acci != 0:
        if step == 0:
            pitches[1:] = np.array([p.alter(-acci) for p in pitches[1:]])
        else:
            pitches[step] = pitches[step].alter(acci)
    return pitches


def _invertPitches(pitches: np.ndarray[OPitch]) -> np.ndarray[OPitch]:
    pitches = -pitches
    pitches[1:] = pitches[1:][::-1]
    return pitches


class Mode(Sequence[OPitch], Set[OPitch], metaclass=ABCMeta):
    """
    A **mode** is a sequence of unique octave intervals in ascending order, starting from
    perfect unison.
    """

    __slots__ = ("_pitches", "_cyc", "_hash")

    if t.TYPE_CHECKING:

        @overload
        def __new__(cls, pitches: Iterable[OPitch | int | str]) -> Self: ...

        @overload
        def __new__(cls, *pitches: OPitch | int | str) -> Self: ...

        @overload
        def __getitem__(self, key: int) -> OPitch: ...

        @overload
        def __getitem__(self, key: slice | Iterable[int]) -> Self:
            """
            Extract a new scale from part of the current scale.
            """
            ...

        @overload
        def alter(self, idx: int, acci: Real) -> Self: ...

        @overload
        def alter(self, idx: Iterable[int], acci: Iterable[Real] | Real) -> Self: ...

        @overload
        def alter(self, acci: Iterable[Real]) -> Self: ...

    def __new__(cls, *args) -> Self:
        if len(args) == 1 and isinstance(args[0], Iterable):
            pitches = args[0]
        else:
            pitches = args
        pitches = np.array([OPitch(p) for p in pitches])
        if pitches[0] != OPitch.C:
            pitches -= pitches[0]
        pitches.sort()
        return cls._newHelper(pitches)

    @classmethod
    def _newHelper(cls, pitches: np.ndarray[OPitch]) -> Self:
        self = object.__new__(cls)
        self._pitches = pitches
        self._pitches.flags.writeable = False
        return self

    def __len__(self) -> int:
        return len(self.pitches)

    def __contains__(self, value: Any) -> bool:
        # a scale is an ordered sequence
        # so use binary search
        if not isinstance(value, PitchBase):
            return False
        idx = bisect_left(self, value)
        return idx < len(self) and self[idx] == value

    @property
    def pitches(self) -> Sequence[OPitch]:
        return self._pitches

    @property
    def cyc(self) -> _ModeCyclicAccessor:
        """Cyclic slicing and access support."""
        return _ModeCyclicAccessor(self)

    def diff(self) -> Iterable[OPitch]:
        """
        Returns the interval structure of the scale, i.e., the differences between adjacent
        pitches.
        """
        return np.diff(self.pitches, append=OPitch.C)

    def __str__(self) -> str:
        return f"({', '.join(map(str, self.pitches))})"

    def __repr__(self) -> str:
        return f"Mode{str(self)}"

    def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Self:
        if isinstance(key, slice):  # generate a new scale by slicing
            return self._slice(self, key)[0]
        elif isinstance(key, Iterable):  # generate a new scale by a set of indices
            return self._multiIndex(self, key)[0]
        else:  # get a pitch by index
            return self._getItem(key)

    def _getItem(self, key: int) -> OPitch:
        # getting a single item
        return self.pitches[key]

    def _slice(self, key: slice) -> tuple[Mode, OPitch]:
        # create a new mode from slicing
        start, _, step = key.indices(len(self))
        newPitches = self.pitches[key].copy()
        if len(newPitches) == 0:
            raise IndexError("empty slice cannot make a scale")
        if step < 0:
            newPitches = np.roll(newPitches, 1)
            startPitch = newPitches[0]
            newPitches -= startPitch
            newPitches[1:] *= -1
        else:
            if start > 0:  # not starting from first note
                startPitch = newPitches[0]
                newPitches -= startPitch
            else:
                startPitch = OPitch.C
        return Mode._newHelper(newPitches), startPitch

    def _cycSlice(self, key: slice) -> tuple[Mode, OPitch]:
        # create a new mode from cyclic slicing
        if key.step == 0:
            if key.start is not None and key.stop is not None and key.start >= key.stop:
                raise IndexError("empty slice cannot make a scale")
            return Mode(), OPitch.C
        negStep = key.step is not None and key.step < 0
        if negStep:
            roll = -key.start - 1 if key.start is not None else -1
            key = slice(-1, key.stop, key.step)
        else:
            roll = -key.start if key.start is not None else 0
            key = slice(0, key.stop, key.step)
        newPitches = np.roll(self.pitches, roll)[key].copy()
        if len(newPitches) == 0:
            raise IndexError("empty slice cannot make a scale")
        startPitch = newPitches[0]
        newPitches -= startPitch
        if negStep:
            newPitches[1:] = -newPitches[1:]
        return Mode._newHelper(newPitches), startPitch

    def _multiIndex(self, key: Iterable[int]) -> tuple[Mode, OPitch]:
        # create a new mode from a set of indices
        indices = SortedSet(key)
        if len(indices) == 0:
            raise IndexError("empty set cannot make a scale")
        newPitches = self.pitches[list(indices)].copy()
        if indices[0] > 0:
            startPitch = newPitches[0]
            newPitches -= startPitch
        else:
            startPitch = OPitch.C
        return Mode._newHelper(newPitches), startPitch

    def _cycMultiIndex(self, key: Iterable[int]) -> tuple[Mode, OPitch]:
        # create a new mode from a set of indices in cyclic order
        # the first index provided is regarded as the new tonic
        key = np.array(list(set(key)))
        start = key[0]
        key -= start
        key %= len(self)
        key.sort()
        newPitches = np.roll(self.pitches, -start)[key]
        startPitch = newPitches[0]
        newPitches -= startPitch
        return Mode._newHelper(newPitches), startPitch

    def alter(
        self,
        arg1: int | Iterable[int] | Iterable[Real],
        arg2: Real | Iterable[Real] | None = None,
    ) -> Self:
        """
        Apply alterations to the scale by adjusting the accidentals of specific pitches.
        """
        if arg2 is None:
            if isinstance(arg1, Iterable):
                newPitches = self._pitches.copy()
                for i, acci in enumerate(arg1):
                    _modeAlter(newPitches, i, acci)
            else:
                return self
        else:
            if isinstance(arg1, Iterable):
                if isinstance(arg2, Iterable):
                    newPitches = self._pitches.copy()
                    for i, acci in zip(arg1, arg2):
                        _modeAlter(newPitches, i, acci)
                else:
                    newPitches = self._pitches.copy()
                    for i in arg1:
                        _modeAlter(newPitches, i, arg2)
            else:
                newPitches = self._pitches.copy()
                _modeAlter(newPitches, arg1, arg2)
        return Mode(newPitches)

    def combine(self, other: Self, offset: OPitch = OPitch.C) -> Mode:
        """
        Combine the current scale with another scale shifted by an interval. The resulting scale
        contains all the pitches of the current scale and the second scale's notes shifted by
        the given interval, repeating notes removed and sorted in ascending order.
        """
        return Mode(it.chain(self._pitches[1:], other._pitches + offset))

    def stack(self, offset: OPitch = OPitch.C) -> Mode:
        """
        Similar to `combine`, but the second scale is the current scale itself.
        """
        return self.combine(self, offset)

    def __and__(self, other: Self) -> Mode:
        newPitches = np.intersect1d(self.pitches, other.pitches)
        return Mode._newHelper(newPitches)

    def __or__(self, other: Self) -> Mode:
        return Mode(it.chain(self.pitches[1:], other.pitches[1:]))

    def __neg__(self) -> Self:
        newPitches = _invertPitches(self._pitches)
        return Mode._newHelper(newPitches)

    def __iter__(self) -> Iterator[OPitch]:
        return iter(self.pitches)

    def __reversed__(self) -> Iterator[OPitch]:
        return reversed(self.pitches)

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            self._hash = hash(tuple(self.pitches))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mode):
            return False
        return self._pitches.shape == other._pitches.shape and np.all(
            self._pitches == other._pitches
        )


class _ModeCyclicAccessor:
    """Helper type providing cyclic indexing and slicing for `Mode` objects."""

    __slots__ = ("_parent",)

    if t.TYPE_CHECKING:

        @overload
        def __getitem__(self, key: int) -> OPitch: ...

        @overload
        def __getitem__(self, key: slice | Iterable[int]) -> Mode: ...

    def __new__(cls, parent: Mode):
        return cls._newHelper(parent)

    @classmethod
    @lru_cache
    def _newHelper(cls, parent: Mode) -> Self:
        self = super().__new__(cls)
        self._parent = parent
        return self

    def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Mode:
        if isinstance(key, slice):
            return self._parent._cycSlice(key)[0]
        elif isinstance(key, Iterable):
            return self._parent._cycMultiIndex(key)[0]
        else:
            key %= len(self._parent)
            return self._parent._getItem(key)


class Modes:
    """Common modes in western music."""

    MAJOR = IONIAN = Mode(range(7))
    HARMONIC_MAJOR = MAJOR.alter(5, -1)
    DORIAN = MAJOR.cyc[1:]
    PHRYGIAN = MAJOR.cyc[2:]
    LYDIAN = MAJOR.cyc[3:]
    MIXOLYDIAN = MAJOR.cyc[4:]
    MINOR = AOLIAN = MAJOR.cyc[5:]
    HARMONIC_MINOR = MINOR.alter(6, 1)
    MELODIC_MINOR = HARMONIC_MINOR.alter(5, 1)
    LOCRIAN = MAJOR.cyc[6:]
    MAJOR_PENTATONIC = CN_GONG = Mode(0, 1, 2, 4, 5)
    CN_SHANG = MAJOR_PENTATONIC.cyc[1:]
    CN_JUE = MAJOR_PENTATONIC.cyc[2:]
    CN_ZHI = MAJOR_PENTATONIC.cyc[3:]
    MINOR_PENTATONIC = CN_YU = MAJOR_PENTATONIC.cyc[4:]
    WHOLE_TONE = WHOLE_TONE_SHARP = Mode(
        0, 1, 2, OPitch(3, 1), OPitch(4, 1), OPitch(5, 1)
    )
    WHOLE_TONE_FLAT = Mode(0, 1, 2, OPitch(4, -1), OPitch(5, -1), OPitch(6, -1))
    BLUES = Mode(0, OPitch(2, -1), 3, OPitch(3, 1), 4, OPitch(6, -1))


class Scale(Sequence[OPitch], Set[OPitch]):
    """A **scale** is a sequence of pitches in a specific mode, starting from a tonic."""

    __slots__ = ("_tonic", "_mode", "_cyc", "_pitches")

    if t.TYPE_CHECKING:

        @overload
        def __getitem__(self, key: int) -> OPitch: ...

        @overload
        def __getitem__(self, key: slice | Iterable[int]) -> Self: ...

    def __new__(
        cls, tonic: OPitch | int | str = OPitch.C, mode: Mode = Modes.MAJOR
    ) -> Self:
        if not isinstance(tonic, OPitch):
            tonic = OPitch(tonic)
        return cls._newHelper(tonic, mode)

    @classmethod
    @lru_cache
    def _newHelper(cls, tonic: OPitch, mode: Mode) -> Self:
        self = super().__new__(cls)
        self._tonic = tonic
        self._mode = mode
        return self

    @property
    def tonic(self) -> OPitch:
        return self._tonic

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def pitches(self) -> Sequence[OPitch]:
        if not hasattr(self, "_pitches"):
            self._pitches = self._mode.pitches + self._tonic
            self._pitches.flags.writeable = False
        return self._pitches

    @property
    def cyc(self) -> _ScaleCyclicAccessor:
        """Cyclic slicing and access support."""
        if not hasattr(self, "_cyc"):
            self._cyc = _ScaleCyclicAccessor(self)
        return self._cyc

    def diff(self) -> Sequence[OPitch]:
        """
        Returns the interval structure of the scale, i.e., the differences between adjacent
        pitches.
        """
        return self.mode.diff()

    def __len__(self):
        return len(self.mode)

    def __contains__(self, value: object):
        if isinstance(value, OPitch):
            return (value - self.tonic) in self.mode
        return False

    def __eq__(self, other: object):
        if isinstance(other, Scale):
            return self.tonic == other.tonic and self.mode == other.mode
        return False

    def __add__(self, other: OPitch) -> Self:
        return self.__class__(self.tonic + other, self.mode)

    def __sub__(self, other) -> Self:
        return self.__class__(self.tonic - other, self.mode)

    def __neg__(self) -> Self:
        return self.__class__(self.tonic, -self.mode)

    def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Self:
        if isinstance(key, slice):
            newMode, startPitch = self.mode._slice(key)
            return self.__class__(self.tonic + startPitch, newMode)
        elif isinstance(key, Iterable):
            newMode, startPitch = self.mode._multiIndex(key)
            return self.__class__(self.tonic + startPitch, newMode)
        else:
            return self.tonic + self.mode._getItem(key)

    def __iter__(self) -> Iterator[OPitch]:
        for interval in self.mode:
            yield self.tonic + interval

    def __repr__(self):
        return f"{self.__class__.__name__}{str(self)}"

    def __str__(self) -> str:
        return f"({', '.join(map(str, self))})"

    def __hash__(self):
        return hash((self.tonic, self.mode))


class _ScaleCyclicAccessor:
    """Helper type providing cyclic indexing and slicing for `Scale` objects."""

    __slots__ = ("_parent",)

    if t.TYPE_CHECKING:

        @overload
        def __getitem__(self, key: int) -> OPitch: ...

        @overload
        def __getitem__(self, key: slice | Iterable[int]) -> Scale: ...

    def __new__(cls, parent: Scale):
        return cls._newHelper(parent)

    @classmethod
    @lru_cache
    def _newHelper(cls, parent: Scale) -> Self:
        self = super().__new__(cls)
        self._parent = parent
        return self

    def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Scale:
        if isinstance(key, slice):
            newMode, startPitch = self._parent.mode._cycSlice(key)
            return Scale._newHelper(self._parent.tonic + startPitch, newMode)
        elif isinstance(key, Iterable):
            newMode, startPitch = self._parent.mode._cycMultiIndex(key)
            return Scale._newHelper(self._parent.tonic + startPitch, newMode)
        else:
            key %= len(self._parent.mode)
            return self._parent[key]
