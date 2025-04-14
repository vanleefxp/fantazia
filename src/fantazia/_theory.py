from __future__ import annotations

import itertools as it
import re
import typing as t
from abc import ABCMeta, abstractmethod
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator, Sequence, Set
from functools import lru_cache
from numbers import Integral, Real
from typing import Self, overload, Protocol, Any
from fractions import Fraction as Q

import numpy as np
from sortedcontainers import SortedSet
import pyrsistent as pyr
from bidict import bidict

if t.TYPE_CHECKING:
    import music21 as m21

from .utils import Rounding, RoundMode, bisect_round, rounding as _round, classconst

__all__ = [
    "AcciPref",
    "DegMaps",
    "Qualities",
    "AcciPrefs",
    "Modes",
    "DegBase",
    "Deg",
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


class Qualities:
    """
    Common interval qualities.
    """

    DOUBLY_AUGMENTED = AA = 3
    AUGMENTED = A = 2
    MAJOR = M = 1
    PERFECT = P = 0
    MINOR = m = -1
    DIMINISHED = d = -2
    DOUBLY_DIMINISHED = dd = -3


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

PERFECT_INTERVAL_DEGS = frozenset((0, 3, 4))
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
    {name: i for i, name in enumerate(SOLFEGE_NAMES)} | {"ti": 6}
)
_intervalQualityMap = bidict(
    (
        ("dd", -3),  # doubly diminished
        ("d", -2),  # diminished
        ("m", -1),  # minor
        ("P", 0),  # perfect
        ("M", 1),  # major
        ("A", 2),  # augmented
        ("AA", 3),  # doubly augmented
    )
)

_leadingAlphaRe = re.compile(r"^(?:[a-z]+|[1-7])", re.IGNORECASE)
_trailingOctaveRe = re.compile(r"_[\+\-]?\d+$")
_multipleAugRe = re.compile(r"^A+")
_multipleAugNumRe = re.compile(r"^\[A\*(\d+)\]")
_multipleDimRe = re.compile(r"^d+")
_multipleDimNumRe = re.compile(r"^\[d\*(\d+)\]")

type AcciPref = Callable[[float], int]
"""
Type alias for a function that takes a tone in half-steps and returns the preferred step for 
that tone. The accidental can be later computed by taking the difference between the given
tone and the standard reference tone in C major scale. Some predefined accidental preference
rules can be found in `AcciPrefs`.
"""


def _parseAcci(src: str) -> int:
    if len(src) == 0:  # no accidental (natural)
        return 0
    if src[0] == "[":  # numeric accidental value wrapped in []
        valueSrc = src[1:-1]  # extract content inside the brackets
        if "/" in valueSrc:  # fractional value
            return Q(valueSrc)
        else:  # float value
            acci = float(valueSrc)
            if acci % 1 == 0:
                acci = int(acci)
            return acci
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


def _parseStep(src: str) -> int:
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


def _parseIntervalQuality(src: str) -> int:
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


def _acci2Str(acci: float) -> str:
    if acci == 0:
        return ""
    elif acci > 0:
        if acci % 1 == 0:
            acci = int(acci)
            if acci <= 3:
                return "+" * acci
            else:
                return f"[{acci:+d}]"
        else:
            return f"[{round(acci, 3):+}]"
    else:
        if acci % 1 == 0:
            acci = int(acci)
            if acci >= -3:
                return "-" * abs(acci)
            else:
                return f"[{acci:+d}]"
        else:
            return f"[{round(acci, 3):+}]"


def _parseStepAndAcci(src: str) -> OPitch:
    # separate step and accidental
    stepNameMatch: re.Match[str] = _leadingAlphaRe.search(src)
    if stepNameMatch is None:
        raise ValueError(f"Invalid pitch format: {src}")
    stepName = stepNameMatch.group().lower()
    acciSrc = src[stepNameMatch.end() :]
    step = _parseStep(stepName)
    acci = _parseAcci(acciSrc)

    return step, acci


class AcciPrefs:
    """See `AcciPref` for details."""

    @staticmethod
    def SHARP(tone: float) -> int:
        """
        Always use the lower step and sharp sign when a tone is not a standard tone in C major
        scale, both for standard 12edo pitches and microtonal pitches.

        Examples:

        | input | output | preferred name
        |:-:|:-:|:-|
        | `1` | `0` | C sharp |
        | `3` | `1` | D sharp |
        | `5.5` | `3` | F quarter-tone-sharp |
        | `6` | `3` | F sharp |
        | `6.5` | `3` | F 3-quarter-tones-sharp |
        | `8` | `4` | G sharp |
        | `10` | `5` | A sharp |
        """
        return bisect_right(MAJOR_SCALE_TONES, tone) - 1

    @staticmethod
    def FLAT(tone: float) -> int:
        """
        Always use the upper step and flat sign when a tone is not a standard tone in C major
        scale,both for standard 12edo pitches and microtonal pitches.

        Examples:

        | input | output | preferred name
        |:-:|:-:|:-|
        | `1` | `1` | D flat |
        | `3` | `2` | E flat |
        | `5.5` | `4` | G 3-quarter-tones-flat |
        | `6` | `4` | G flat |
        | `6.5` | `4` | G quarter-tone-flat |
        | `8` | `5` | A flat |
        | `10` | `6` | B flat |
        """
        return bisect_left(MAJOR_SCALE_TONES, tone) - 1

    @staticmethod
    def CLOSEST_SHARP(tone: float) -> int:
        """
        For 12edo pitches, use the lower step and sharp sign when the tone is not a standard tone
        in C major scale. For microtonal pitches, choose the closest standard tone in C major scale.

        Examples:

        | input | output | preferred name
        |:-:|:-:|:-|
        | `1` | `0` | C sharp |
        | `3` | `1` | D sharp |
        | `5.5` | `3` | F quarter-tone-sharp |
        | `6` | `3` | F sharp |
        | `6.5` | `4` | G quarter-tone-flat |
        | `8` | `4` | G sharp |
        | `10` | `5` | A sharp |
        """
        if tone > 11.5:
            return 7
        return bisect_round(MAJOR_SCALE_TONES, tone, roundingMode=RoundMode.HALF_DOWN)

    @staticmethod
    def CLOSEST_FLAT(tone: float) -> int:
        """
        For 12edo pitches, use the upper step and flat sign when the tone is not a standard tone
        in C major scale. For microtonal pitches, choose the closest standard tone in C major scale.

        Examples:

        | input | output | preferred name |
        |:-:|:-:|:-|
        | `1` | `1` | D flat |
        | `3` | `2` | E flat |
        | `5.5` | `3` | F quarter-tone-sharp |
        | `6` | `4` | G flat |
        | `6.5` | `4` | G quarter-tone-flat |
        | `8` | `5` | A flat |
        | `10` | `6` | B flat |
        """
        if tone >= 11.5:
            return 7
        return bisect_round(MAJOR_SCALE_TONES, tone, roundingMode=RoundMode.HALF_UP)

    @staticmethod
    def CLOSEST_FLAT_F_SHARP(tone: float) -> int:
        """
        Same as `CLOSEST_FLAT`, but for the tritone (`tone == 6`) case, use the F sharp instead of
        G flat.

        Examples:

        | input | output | preferred name |
        |:-:|:-:|:-|
        | `1` | `1` | D flat |
        | `3` | `2` | E flat |
        | `5.5` | `3` | F quarter-tone-sharp |
        | `6` | `3` | F sharp |
        | `6.5` | `4` | G quarter-tone-flat |
        | `8` | `5` | A flat |
        | `10` | `6` | B flat |
        """
        if tone >= 11.5:
            return 7
        step = bisect_round(MAJOR_SCALE_TONES, tone, roundingMode=RoundMode.HALF_UP)
        if tone == 6:
            return 3
        else:
            return step

    def __init__(self):
        raise TypeError("This class is not intended to be instantiated")


class SupportsGetItem(Protocol):
    def __getitem__(self, key: int) -> Self: ...


class DegBase(metaclass=ABCMeta):
    @property
    @abstractmethod
    def step(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def acci(self) -> float:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"

    def __str__(self):
        return f"{self.step + 1}{_acci2Str(self.acci)}"


class Deg(DegBase):
    __slots__ = ("_step", "_acci")

    if t.TYPE_CHECKING:

        @overload
        def __new__(cls, src: str): ...

        @overload
        def __new__(cls, step: int, acci: Real): ...

    def __new__(cls, arg1: str | int, arg2: Real | str | None = None) -> Self:
        if arg2 is None:
            if isinstance(arg1, cls):
                return arg1
            elif isinstance(arg1, DegBase):
                step, acci = arg1.step % 7, arg1.acci
                return cls._newHelper(step, acci)
            elif isinstance(arg1, str):
                return cls._parseStr(arg1)
            arg2 = 0

        if isinstance(arg1, str):
            step = _parseStep(arg1)
        else:
            step = arg1 % 7
        if isinstance(arg2, str):
            acci = _parseAcci(arg2)
        else:
            acci = arg2
            if acci % 1 == 0:
                acci = int(acci)
        return cls._newHelper(step, acci)

    @classmethod
    @lru_cache
    def _parseStr(cls, src: str) -> Self:
        return cls._newHelper(*_parseStepAndAcci(src))

    @classmethod
    @lru_cache
    def _newHelper(cls, step: int, acci: float) -> Self:
        # create a new instance with caching
        return cls._newImpl(step, acci)

    @classmethod
    def _newImpl(cls, step: int, acci: float) -> Self:
        # the implementation of creating a new instance without caching
        self = super(DegBase, cls).__new__(cls)
        self._step = step
        self._acci = acci
        return self

    @property
    def step(self) -> int:
        """The scale step of degree, ranging from 0 to 6"""
        return self._step

    @property
    def acci(self) -> float:
        """An accidental value in half-steps that modifies the degree."""
        return self._acci


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
        self, key: int | str | Deg | slice | Iterable[int]
    ) -> int | Sequence[int]:
        if isinstance(key, tuple):
            key = np.array(key)
        if isinstance(key, Integral):
            return self._tones[key]
        else:
            key = Deg(key)
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
        return map(OPitch.fromStepAndTone, range(7), self._parent._tones)

    def __contains__(self, value: Any):
        if not isinstance(value, PitchBase):
            return False
        return self._parent._tones[value.step % 7] == value.tone

    def index(self, value: PitchBase) -> Deg:
        """Find the degree of pitch in the degree map."""
        value = value.opitch
        step = value.step
        acci = value.tone - self._parent._tones[step]
        return Deg(step, acci)

    def __eq__(self, other) -> bool:
        return self._parent == other._parent

    def __getitem__(self, key: int) -> int:
        if isinstance(key, slice):
            return self._slice(key)
        if isinstance(key, Iterable) and not isinstance(key, str):
            return self._multiIndex(key)
        return self._getItem(key)

    def _getItem(self, key: int | str | Deg) -> OPitch:
        if isinstance(key, Integral):
            return OPitch.fromStepAndTone(key, self._parent._tones[key])
        if not isinstance(key, Deg):
            key = Deg(key)
        return OPitch.fromStepAndTone(
            key.step, self._parent._tones[key.step] + key.acci
        )

    def _slice(self, key: slice) -> Sequence[OPitch]:
        steps = np.arange(7)[key]
        tones = self._parent._tones[steps]
        return np.array(
            [OPitch.fromStepAndTone(s, t) for s, t in zip(steps, tones)], dtype=object
        )

    def _multiIndex(self, key: Iterable[int | str | Deg]) -> Sequence[OPitch]:
        return np.array([self._getItem(k) for k in key], dtype=object)


class PitchBase(DegBase):
    """Base class for `OPitch` and `Pitch`."""

    @property
    @abstractmethod
    def step(self) -> int:
        """Degree of the pitch in C major scale, not considering accidentals."""
        raise NotImplementedError

    @property
    @abstractmethod
    def acci(self) -> Real:
        """Accidental of the pitch, in half-steps."""
        raise NotImplementedError

    @property
    def tone(self) -> Real:
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
    def quality(self) -> float:
        """
        Interval quality when regarding the pitch as an interval.
        """
        return self.opitch.quality

    @property
    def freq(self) -> float:
        """
        Frequency of the pitch relative to middle C.
        """
        return np.pow(2, self.tone / 12)

    @abstractmethod
    def withAcci(self, acci: float = 0) -> Self:
        """
        Returns a new instance of the same class with the given accidental.
        """
        raise NotImplementedError

    def alter(self, acci: float = 0) -> Self:
        return self.withAcci(self.acci + acci)

    @abstractmethod
    def atOctave(self, octave: int = 0) -> Pitch:
        raise NotImplementedError

    def isEnharmonic(self, other: Self) -> bool:
        return self.tone == other.tone

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        raise NotImplementedError

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
        useNatural: bool = False,
        useQuartertone: bool = True,
        rounding: Rounding = Rounding.ROUND,
        roundMode: RoundMode = RoundMode.HALF_EVEN,
        asInterval: bool = False,
    ) -> m21.pitch.Pitch:
        """Convert to a `music21.pitch.Pitch` object."""
        try:
            import music21 as m21
        except ImportError:
            raise ImportError("`music21` must be installed first for the conversion.")

        m21_step = STEP_NAMES[self.step % 7]
        m21_acci = self.acci
        m21_acci, m21_microtone = _round(
            m21_acci, 0.5 if useQuartertone else 1, rounding, roundMode
        )
        m21_microtone *= 100
        if not useNatural and m21_acci == 0:
            m21_acci = None

        return m21.pitch.Pitch(
            step=m21_step,
            accidental=m21_acci,
            microtone=m21_microtone,
            octave=self.o + 4 if isinstance(self, Pitch) else None,
        )


class OPitch(PitchBase, Deg):
    """
    Represents a pitch in an octave, or equivalently, an interval no greater than an octave.
    """

    __slots__ = ("_step", "_acci")
    _C = None

    if t.TYPE_CHECKING:

        @overload
        def __new__(cls, src: str) -> Self:
            """
            Creates a new `OPitch` object from string notation. A valid string notation is a
            note name followed by optional accidental symbols. Sharps and flats ar represented
            by `+` and `-` symbols, respectively. `++` and `--` are used for double sharps and
            double flats. For microtonal notations, the accidental is a float value preceded by
            `+` of `-` wrapped in brackets.
            """
            ...

        @overload
        def __new__(cls, step: int | str = 0, acci: Real | str = 0) -> Self:
            """
            Creates a new `OPitch` object from step and accidental values.
            """
            ...

        @overload
        def __new__(cls, src: PitchBase) -> Self: ...

    @classconst
    def C(cls) -> Self:
        """
        The C pitch, with no accidental. This is the identity element of addition in the octave
        pitch abelian group.
        """
        if cls._C is None:
            cls._C = cls._newHelper(0, 0)
        return cls._C

    @classmethod
    def fromStepAndTone(cls, step: int | str, tone: Real) -> Self:
        """
        Creates a pitch from a step and a chromatic tone.
        """
        if isinstance(step, str):
            step = _parseStep(step)
        if not isinstance(step, Integral):
            raise TypeError(f"invalid step: {step}")
        octave, step = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[step] - octave * 12
        return cls(step, acci)

    @classmethod
    def fromTone(
        cls,
        tone: Real | PitchBase,
        acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP,
    ) -> Self:
        if isinstance(tone, PitchBase):
            tone = tone.tone
        tone %= 12
        step = acciPref(tone)
        octaves, ostep = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[ostep] - octaves * 12
        return cls(step, acci)

    @classmethod
    def co5(cls, n: int = 0) -> Self:
        """
        Returns the `n`-th pitch in circle of fifths order, starting from C.
        positive `n` values means `n` perfect fifths up while negative `n` values means `n`
        perfect fifths down.

        When `n` ranges from -7 to 7, this method yields the tonic of major scale with `abs(n)`
        sharps (for positive `n`) or flats (for negative `n`).
        """
        step = n * 4 % 7
        acci = (n + 1) // 7
        return cls(step, acci)

    def __new__(
        cls, arg1: int | str | PitchBase = 0, arg2: int | float | Q | str | None = None
    ) -> Self:
        if arg2 is None and isinstance(arg1, str):
            return cls._parseStr(arg1)
        if isinstance(arg1, PitchBase):
            if arg2 is None:
                return arg1.opitch
            else:
                step = arg1.step
                acci = arg1.acci + arg2
                return cls._newHelper(step, acci)

        if isinstance(arg1, str):
            step = _parseStep(arg1)
        else:
            step = arg1 % 7

        if arg2 is None:
            acci = 0
        elif isinstance(arg2, str):
            acci = _parseAcci(arg2)
        else:
            acci = arg2
            if acci % 1 == 0:
                acci = int(acci)

        return cls._newHelper(step, acci)

    @property
    def step(self) -> int:
        return self._step

    @property
    def acci(self) -> float:
        return self._acci

    @property
    def opitch(self) -> OPitch:
        return self

    @property
    def o(self) -> int:
        return 0

    @property
    def tone(self) -> float:
        """
        Chromatic tone of the pitch, in half-steps.
        """
        return MAJOR_SCALE_TONES[self.step] + self.acci

    @property
    def quality(self) -> float:
        if self.acci > 0:
            return self.acci + 1
        elif self.step in PERFECT_INTERVAL_DEGS:
            if self.acci == 0:
                return 0
            else:
                return self.acci - 1
        else:
            if self.acci == 0:
                return 1
            else:
                return self.acci

    def atOctave(self, octave: int = 0) -> Pitch:
        return Pitch._newHelper(self, octave - self.tone // 12)

    def withAcci(self, acci: float = 0) -> Self:
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

    def __str__(self) -> str:
        return f"{STEP_NAMES[self.step]}{_acci2Str(self.acci)}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    def __hash__(self) -> int:
        return hash((self.step, self.acci))


class Pitch(PitchBase):
    """
    Represents a pitch with specific octave, or an interval that may cross multiple octaves.
    """

    __slots__ = ("_opitch", "_o")
    _C0 = None

    if t.TYPE_CHECKING:

        @overload
        def __new__(cls, src: str) -> Self: ...

        @overload
        def __new__(cls, step: int = 0, acci: float = 0) -> Self: ...

        @overload
        def __new__(cls, opitch: PitchBase = OPitch.C, octave: int = 0) -> Self: ...

    @classconst
    def C0(cls) -> Self:
        """
        The middle C pitch. It is the identity value of the pitch abelian group.
        """
        if cls._C0 is None:
            cls._C0 = cls._newHelper(OPitch.C, 0)
        return cls._C0

    @classmethod
    def fromTone(cls, tone: float, acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP):
        opitch = OPitch.fromTone(tone, acciPref)
        octave = (tone - opitch.tone) // 12
        return cls._newHelper(opitch, octave)

    def __new__(cls, arg1: int | PitchBase, arg2: float | int | None = None) -> Self:
        if isinstance(arg1, str):
            return cls._parseStr(arg1)
        if isinstance(arg1, Pitch):
            if arg2 is None:
                return arg1
            arg1 = arg1.opitch
        if isinstance(arg1, OPitch):
            opitch = arg1
            octave = arg2
        else:
            octave, ostep = divmod(arg1, 7)
            opitch = OPitch(ostep, arg2)
        return cls._newHelper(opitch, octave)

    @classmethod
    @lru_cache
    def _parseStr(self, src: str) -> Self:
        if not src[-1].isdigit():
            # octave not specified, assume octave 0
            opitch = OPitch._parseStr(src)
            octave = 0
        else:
            match: re.Match[str] = _trailingOctaveRe.search(src)
            opitch = OPitch._parseStr(src[: match.start()])
            octave = int(match.group()[1:])
        return Pitch._newHelper(opitch, octave)

    @classmethod
    @lru_cache
    def _newHelper(cls, opitch: OPitch, octave: int) -> Self:
        return cls._newImpl(opitch, octave)

    @classmethod
    def _newImpl(cls, opitch: OPitch, octave: int) -> Self:
        self = super().__new__(cls)
        self._opitch = opitch
        self._o = octave
        return self

    @property
    def opitch(self) -> OPitch:
        return self._opitch

    @property
    def otone(self) -> float:
        return self.opitch.otone

    @property
    def o(self) -> int:
        return self._o

    @property
    def step(self) -> int:
        """
        Degree of the pitch in C major scale, not considering accidentals. Equals to the step
        of octave pitch plus nominal octave times 7.
        """
        return self.opitch.step + self.o * 7

    @property
    def acci(self) -> float:
        return self.opitch.acci

    @property
    def tone(self) -> float:
        """
        Chromatic tone of the pitch, in half-steps. Equals to the chromatic tone of octave
        pitch plus nominal octave times 12.
        """
        return self.opitch.tone + self.o * 12

    @property
    def quality(self) -> float:
        return self.opitch.quality

    def withAcci(self, acci: float = 0) -> Self:
        return self._newHelper(self.opitch.withAcci(acci), self.o)

    def atOctave(self, octave: int = 0) -> Self:
        return self.opitch.atOctave(octave)

    def hz(self, ref: float = 440) -> float:
        return ref * np.power(2, (self.tone - 9) / 12)

    def __add__(self, other: PitchBase) -> Self:
        step = self.step + other.step
        tone = self.tone + other.tone
        octave, ostep = divmod(step, 7)
        acci = tone - MAJOR_SCALE_TONES[ostep] - octave * 12
        return OPitch(ostep, acci).atOctave(octave)

    def __neg__(self) -> Self:
        if self.opitch.step == 0:
            return (-self.opitch).atOctave(-self.o)
        return (-self.opitch).atOctave(-self.o - 1)

    def __mul__(self, other: int) -> Self:
        if other == 0:
            return self.__class__()
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
        return OPitch(ostep, acci).atOctave(octave)

    def __str__(self):
        return f"{STEP_NAMES[self.opitch.step]}{_acci2Str(self.acci)}_{self.o}"

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"

    def __hash__(self) -> int:
        return hash((self.opitch, self.o))


def _modeAlter(
    pitches: np.ndarray[OPitch], step: int, acci: float
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
        if not hasattr(self, "_cyc"):
            self._cyc = _ModeCyclicAccessor(self)
        return self._cyc

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
        arg1: int | Iterable[int] | Iterable[float],
        arg2: float | Iterable[float] | None = None,
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

    @overload
    def __getitem__(self, key: int) -> OPitch: ...

    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Self: ...

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

    def __new__(cls, parent: Scale):
        return cls._newHelper(parent)

    @classmethod
    @lru_cache
    def _newHelper(cls, parent: Scale) -> Self:
        self = super().__new__(cls)
        self._parent = parent
        return self

    @overload
    def __getitem__(self, key: int) -> OPitch: ...

    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Scale: ...

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
