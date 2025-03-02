from collections.abc import Callable, Sequence, Set, Iterable, Iterator
from typing import Self, overload
from numbers import Integral
from abc import ABCMeta, abstractmethod
from bisect import bisect_left, bisect_right
import itertools as it
from functools import lru_cache

import numpy as np
from sortedcontainers import SortedSet

from .utils import bisect_round, RoundingMode

__all__ = [
    "AcciPref",
    "Accis",
    "Degs",
    "Intervals",
    "Qualities",
    "AcciPrefs",
    "Modes",
    "PitchLike",
    "OPitch",
    "Pitch",
    "Mode",
    "OInterval",
    "Interval",
    "OIntervalSet",
]


class Accis:
    """
    Common Accidental constants.
    """

    TRIPLE_SHARP = SSS = 3
    DOUBLE_SHARP = SS = 2
    SHARP = S = 1
    NATURAL = N = 0
    FLAT = F = -1
    DOOUBLE_FLAT = FF = -2
    TRIPLE_FLAT = FFF = -3


class Degs:
    """
    Degree constants. Supports both letter names and fixed-do solfege notations.
    """

    DO = C = 0
    RE = D = 1
    MI = E = 2
    FA = F = 3
    SOL = G = 4
    LA = A = 5
    SI = B = 6


class Intervals:
    """
    Common intervals.
    """

    UNISON = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    FIFTH = 4
    SIXTH = 5
    SEVENTH = 6


class Qualities:
    """
    Common interval qualities.
    """

    DOUBLY_AUGMENTED = 3
    AUGMENTED = 2
    MAJOR = 1
    PERFECT = 0
    MINOR = -1
    DIMINISHED = -2
    DOUBLY_DIMINISHED = -3


# major scale notes in circle of fifths order
_majorScaleCo5Order = np.arange(-1, 6) * 7 % 12
_majorScaleCo5Order.flags.writeable = False

# major scale notes in increasing order
_majorScale = np.sort(_majorScaleCo5Order)
_majorScale.flags.writeable = False

_degreeMask = np.full(12, -1, dtype=int)
_degreeMask[_majorScale] = np.arange(7)
_degreeMask.flags.writeable = False

_perfectIntervals = frozenset((0, 3, 4))
_noteNames = np.roll(np.array([chr(65 + i) for i in range(7)]), -2)

type AcciPref = Callable[[float], int]
"""
Type alias for a function that takes a tone in half-steps and returns the preferred degree for 
that tone. The accidental can be later computed by taking the difference between the given
tone and the standard reference tone in C major scale. Some predefined accidental preference
rules can be found in `AcciPrefs`.
"""


def _accidentalToStr(acci: float) -> str:
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
            return f"[{acci:+.3f}]"
    else:
        if acci % 1 == 0:
            acci = int(acci)
            if acci >= -3:
                return "-" * abs(acci)
            else:
                return f"[{acci:+d}]"
        else:
            return f"[{acci:+.3f}]"


def _alwaysSharp(tone: float) -> int:
    return bisect_right(_majorScale, tone) - 1


def _alwaysFlat(tone: float) -> int:
    return bisect_left(_majorScale, tone)


def _closestSharp(tone: float) -> int:
    if tone > 11.5:
        return 7
    return bisect_round(_majorScale, tone, roundingMode=RoundingMode.HALF_DOWN)


def _closestFlat(tone: float) -> int:
    if tone >= 11.5:
        return 7
    return bisect_round(_majorScale, tone, roundingMode=RoundingMode.HALF_UP)


def _closestFlatFSharp(tone: float) -> int:
    if tone >= 11.5:
        return 7
    deg = bisect_round(_majorScale, tone, roundingMode=RoundingMode.HALF_UP)
    if tone == 6:
        return 3
    else:
        return deg


class AcciPrefs:
    """See `AcciPref` for details."""

    SHARP = _alwaysSharp
    """
    Always use the lower degree and sharp sign when a tone is not a standard tone in C major 
    scale, both for standard 12edo pitches and microtonal pitches.
    
    Examples:
    
    | input `tone` | output `degree` | preferred name
    |:-:|:-:|:-|
    | `1` | `0` | C sharp |
    | `3` | `1` | D sharp |
    | `5.5` | `3` | F quarter-tone-sharp |
    | `6` | `3` | F sharp |
    | `6.5` | `3` | F 3-quarter-tones-sharp |
    | `8` | `4` | G sharp |
    | `10` | `5` | A sharp |
    """

    FLAT = _alwaysFlat
    """
    Always use the upper degree and flat sign when a tone is not a standard tone in C major 
    scale,both for standard 12edo pitches and microtonal pitches.
    
    Examples:
    
    | input `tone` | output `degree` | preferred name
    |:-:|:-:|:-|
    | `1` | `1` | D flat |
    | `3` | `2` | E flat |
    | `5.5` | `4` | G 3-quarter-tones-flat |
    | `6` | `4` | G flat |
    | `6.5` | `4` | G quarter-tone-flat |
    | `8` | `5` | A flat |
    | `10` | `6` | B flat |
    """

    CLOSEST_SHARP = _closestSharp
    """
    For 12edo pitches, use the lower degree and sharp sign when the tone is not a standard tone 
    in C major scale. For microtonal pitches, choose the closest standard tone in C major scale.
    
    Examples:
    
    | input `tone` | output `degree` | preferred name
    |:-:|:-:|:-|
    | `1` | `0` | C sharp |
    | `3` | `1` | D sharp |
    | `5.5` | `3` | F quarter-tone-sharp |
    | `6` | `3` | F sharp |
    | `6.5` | `4` | G quarter-tone-flat |
    | `8` | `4` | G sharp |
    | `10` | `5` | A sharp |
    """

    CLOSEST_FLAT = _closestFlat
    """
    For 12edo pitches, use the upper degree and flat sign when the tone is not a standard tone 
    in C major scale. For microtonal pitches, choose the closest standard tone in C major scale.
    
    Examples:
    
    | input `tone` | output `degree` | preferred name |
    |:-:|:-:|:-|
    | `1` | `1` | D flat |
    | `3` | `2` | E flat |
    | `5.5` | `3` | F quarter-tone-sharp |
    | `6` | `4` | G flat |
    | `6.5` | `4` | G quarter-tone-flat |
    | `8` | `5` | A flat |
    | `10` | `6` | B flat |
    """

    CLOSEST_FLAT_F_SHARP = _closestFlatFSharp
    """
    Same as `CLOSEST_FLAT`, but for the tritone (`tone == 6`) case, use the F sharp instead of 
    G flat. 
    
    Examples:
    
    | input `tone` | output `degree` | preferred name |
    |:-:|:-:|:-|
    | `1` | `1` | D flat |
    | `3` | `2` | E flat |
    | `5.5` | `3` | F quarter-tone-sharp |
    | `6` | `3` | F sharp |
    | `6.5` | `4` | G quarter-tone-flat |
    | `8` | `5` | A flat |
    | `10` | `6` | B flat |
    """


class PitchLike(metaclass=ABCMeta):
    """Base class for `OPitch` and `Pitch`."""

    @property
    @abstractmethod
    def deg(self) -> int:
        """Degree of the pitch in C major scale, not considering accidentals."""
        raise NotImplementedError

    @property
    @abstractmethod
    def acci(self) -> float:
        """Accidental of the pitch, in half-steps."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tone(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def quality(self) -> float:
        """
        Interval quality when regarding the pitch as an interval.
        """
        raise NotImplementedError

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
    def atOctave(self, octave: int = 0) -> "Pitch":
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
            self.tone == other.tone and self.deg < other.deg
        )

    def __eq__(self, other: Self):
        return self.deg == other.deg and self.acci == other.acci


class OPitch(PitchLike):
    """
    Represents a pitch in an octave, or equivalently, an interval no greater than an octave.
    """

    @classmethod
    def fromDegAndTone(cls, deg: int, tone: float) -> Self:
        """
        Creates a pitch from a degree and a chromatic tone.
        """
        octave, deg = divmod(deg, 7)
        acci = tone - _majorScale[deg] - octave * 12
        return cls(deg, acci)

    @classmethod
    def fromTone(
        cls,
        tone: float,
        acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP,
    ) -> Self:
        tone %= 12
        deg = acciPref(tone)
        octaves, odeg = divmod(deg, 7)
        acci = tone - _majorScale[odeg] - octaves * 12
        return cls(deg, acci)

    def __new__(cls, deg: int | str = 0, acci: float = 0) -> Self:
        if isinstance(deg, str):
            deg = ord(deg.upper()) - 67
        if not isinstance(deg, Integral):
            raise TypeError(f"degree must be an integer, got {deg.__class__.__name__}")
        if acci % 1 == 0:
            acci = int(acci)
        deg %= 7
        return cls._newHelper(deg, acci)

    @classmethod
    @lru_cache
    def _newHelper(cls, deg: int, acci: float) -> Self:
        self = super().__new__(cls)
        self._deg = deg
        self._acci = acci
        return self

    @property
    def deg(self) -> int:
        return self._deg

    @property
    def acci(self) -> float:
        return self._acci

    @classmethod
    def co5(cls, n: int = 0) -> Self:
        """
        Returns the `n`-th pitch in circle of fifths order, starting from C.
        positive `n` values means `n` perfect fifths up while negative `n` values means `n`
        perfect fifths down.

        When `n` ranges from -7 to 7, this method yields the tonic of major scale with `abs(n)`
        sharps (for positive `n`) or flats (for negative `n`).
        """
        deg = n * 4 % 7
        acci = (n + 1) // 7
        return cls(deg, acci)

    @property
    def tone(self) -> float:
        """
        Chromatic tone of the pitch, in half-steps.
        """
        return _majorScale[self.deg] + self.acci

    @property
    def quality(self) -> float:
        if self.acci > 0:
            return self.acci + 1
        elif self.deg in _perfectIntervals:
            if self.acci == 0:
                return 0
            else:
                return self.acci - 1
        else:
            if self.acci == 0:
                return 1
            else:
                return self.acci

    def atOctave(self, octave: int = 0) -> "Pitch":
        return Pitch._newHelper(self, octave - self.tone // 12)

    def withAcci(self, acci: float = 0) -> Self:
        return self.__class__(self.deg, acci)

    def isEnharmonic(self, other: Self) -> bool:
        return (self.tone - other.tone) % 12 == 0

    def __add__(self, other: Self) -> Self:
        deg = self.deg + other.deg
        octave, deg = divmod(deg, 7)
        tone = self.tone + other.tone
        acci = tone - _majorScale[deg] - octave * 12 + self.acci
        return self.__class__(deg, acci)

    def __neg__(self) -> Self:
        if self.deg == 0:
            return self.__class__(0, -self.acci)
        deg = 7 - self.deg
        tone = 12 - self.tone
        acci = tone - _majorScale[deg]
        return self.__class__(deg, acci)

    def __mul__(self, other: int) -> Self:
        if other == 0:
            return self.__class__()
        if not isinstance(other, Integral):
            raise TypeError(f"can only multiply by integers, got {type(other)}")
        deg = self.deg * other
        tone = self.tone * other
        octave, deg = divmod(deg, 7)
        acci = tone - _majorScale[deg] - octave * 12 + self.acci
        return self.__class__(deg, acci)

    def __str__(self) -> str:
        return f"{_noteNames[self.deg]}{_accidentalToStr(self.acci)}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    def __hash__(self) -> int:
        return hash((self.deg, self.acci))


class Pitch(PitchLike):
    """
    Represents a pitch with specific octave, or an interval that may cross multiple octaves.
    """

    __slots__ = ("_opitch", "_octave")

    @classmethod
    def fromTone(cls, tone: float, acciPref: AcciPref = AcciPrefs.CLOSEST_FLAT_F_SHARP):
        opitch = OPitch.fromTone(tone, acciPref)
        octave = (tone - opitch.tone) // 12
        return cls._newHelper(opitch, octave)

    @overload
    def __new__(cls, deg: int = 0, acci: float = 0) -> Self: ...

    @overload
    def __new__(cls, opitch: PitchLike = OPitch(), octave: int = 0) -> Self: ...

    def __new__(cls, arg1: int | PitchLike, arg2: float | int) -> Self:
        if isinstance(arg1, Pitch):
            arg1 = arg1.opitch
        if isinstance(arg1, OPitch):
            opitch = arg1
            octave = arg2
        else:
            octave, odeg = divmod(arg1, 7)
            opitch = OPitch(odeg, arg2)
        return cls._newHelper(opitch, octave)

    @classmethod
    @lru_cache
    def _newHelper(cls, opitch: OPitch, octave: int) -> Self:
        self = super().__new__(cls)
        self._opitch = opitch
        self._octave = octave
        return self

    @property
    def opitch(self) -> OPitch:
        return self._opitch

    @property
    def octave(self) -> int:
        return self._octave

    @property
    def deg(self) -> int:
        """
        Degree of the pitch in C major scale, not considering accidentals. Equals to the degree
        of octave pitch plus octave times 7.
        """
        return self.opitch.deg + self.octave * 7

    @property
    def acci(self) -> float:
        return self.opitch.acci

    @property
    def tone(self) -> float:
        """
        Chromatic tone of the pitch, in half-steps. Equals to the chromatic tone of octave
        pitch plus octave times 12.
        """
        return self.opitch.tone + self.octave * 12

    @property
    def quality(self) -> float:
        return self.opitch.quality

    def withAcci(self, acci: float = 0) -> Self:
        return self.opitch.withAcci(acci).atOctave(self.octave)

    def atOctave(self, octave: int = 0) -> Self:
        return self.opitch.atOctave(octave)

    def hz(self, A0: float = 440) -> float:
        return A0 * np.power(2, (self.tone - 9) / 12)

    def __add__(self, other: PitchLike) -> Self:
        deg = self.deg + other.deg
        tone = self.tone + other.tone
        octave, odeg = divmod(deg, 7)
        acci = tone - _majorScale[odeg] - octave * 12 + self.acci
        return OPitch(odeg, acci).atOctave(octave)

    def __neg__(self) -> Self:
        if self.opitch.deg == 0:
            return (-self.opitch).atOctave(-self.octave)
        return (-self.opitch).atOctave(-self.octave - 1)

    def __mul__(self, other: int) -> Self:
        if other == 0:
            return self.__class__()
        if not isinstance(other, Integral):
            raise TypeError(
                f"can only multiply by integers, got {other.__class__.__name__}"
            )
        deg = self.deg * other
        tone = self.tone * other
        octave, odeg = divmod(deg, 7)
        acci = tone - _majorScale[odeg] - octave * 12 + self.acci
        return OPitch(odeg, acci).atOctave(octave)

    def __str__(self):
        return (
            f"{_noteNames[self.opitch.deg]}{self.octave}{_accidentalToStr(self.acci)}"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"

    def __hash__(self) -> int:
        return hash((self.opitch, self.octave))


def _scaleAlter(
    pitches: np.ndarray[OPitch], deg: int, acci: float
) -> np.ndarray[OPitch]:
    if acci != 0:
        if deg == 0:
            pitches[1:] = np.array([p.alter(-acci) for p in pitches[1:]])
        else:
            pitches[deg] = pitches[deg].alter(acci)
    return pitches


def _scaleInvert(pitches: np.ndarray[OPitch]) -> np.ndarray[OPitch]:
    pitches = -pitches
    pitches[1:] = pitches[1:][::-1]
    return pitches


class Mode(Sequence[OPitch], Set[OPitch]):
    """
    A **mode** is a sequence of unique octave intervals in ascending order, starting from
    perfect unison.
    """

    __slots__ = ("_pitches", "_cyc")

    @classmethod
    def _newFromTrustedArray(cls, pitches: np.ndarray[OPitch]) -> Self:
        self = super().__new__(cls)
        self._pitches = pitches
        self._pitches.flags.writeable = False
        return self

    @overload
    def __new__(cls, pitches: Iterable[OPitch | int]) -> Self: ...

    @overload
    def __new__(cls, *pitches: OPitch | int) -> Self: ...

    def __new__(cls, *args) -> Self:
        if len(args) == 1 and isinstance(args[0], Iterable):
            pitches = args[0]
        else:
            pitches = args
        pitches = (p if isinstance(p, OPitch) else OPitch(p) for p in pitches)
        return cls._newFromTrustedArray(
            np.array(
                SortedSet(it.chain((OPitch(),), pitches)),
                dtype=object,
            )
        )

    @overload
    def __getitem__(self, key: int) -> OPitch: ...

    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Self:
        """
        Extract a new scale from part of the current scale.
        """
        ...

    def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Self:
        if isinstance(key, slice):  # generate a new scale by slicing
            start, stop, step = key.indices(len(self))
            newPitches = self._pitches[key].copy()
            if len(newPitches) == 0:
                raise IndexError("empty slice cannot make a scale")
            if step < 0:
                print(newPitches)
                newPitches = np.roll(newPitches, 1)
                newPitches -= newPitches[0]
                newPitches[1:] *= -1
            else:
                if start > 0:  # not starting from first note
                    newPitches -= newPitches[0]
            return self._newFromTrustedArray(newPitches)
        elif isinstance(key, Iterable):  # generate a new scale by a set of indices
            indices = SortedSet(key)
            if len(indices) == 0:
                raise IndexError("empty set cannot make a scale")
            newPitches = self._pitches[list(indices)].copy()
            if indices[0] > 0:
                newPitches -= newPitches[0]
            return self._newFromTrustedArray(newPitches)
        else:  # get a pitch by index
            return self._pitches[key]

    def __len__(self) -> int:
        return len(self._pitches)

    def __iter__(self) -> Iterator[OPitch]:
        return iter(self._pitches)

    @property
    def cyc(self) -> "_ModeCyclicAccessor":
        """Cyclic slicing and access support."""
        if not hasattr(self, "_cyc"):
            self._cyc = _ModeCyclicAccessor(self)
        return self._cyc

    def roll(self, n: int) -> Self:
        newPitches = np.roll(self._pitches, n)
        newPitches -= newPitches[0]
        return self._newFromTrustedArray(newPitches)

    def diff(self) -> Sequence[OPitch]:
        """
        Returns the interval structure of the scale, i.e., the differences between adjacent
        pitches.
        """
        return np.diff(self._pitches, append=OPitch())

    @overload
    def alter(self, idx: int, acci: float) -> Self: ...

    @overload
    def alter(self, idx: Iterable[int], acci: Iterable[float] | float) -> Self: ...

    @overload
    def alter(self, acci: Iterable[float]) -> Self: ...

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
                    _scaleAlter(newPitches, i, acci)
            else:
                return self
        else:
            if isinstance(arg1, Iterable):
                if isinstance(arg2, Iterable):
                    newPitches = self._pitches.copy()
                    for i, acci in zip(arg1, arg2):
                        _scaleAlter(newPitches, i, acci)
                else:
                    newPitches = self._pitches.copy()
                    for i in arg1:
                        _scaleAlter(newPitches, i, arg2)
            else:
                newPitches = self._pitches.copy()
                _scaleAlter(newPitches, arg1, arg2)
        return self.__class__(newPitches)

    def __contains__(self, value) -> bool:
        # a scale is an ordered sequence
        # so use binary search
        if not isinstance(value, PitchLike):
            return False
        idx = bisect_left(self._pitches, value)
        if idx >= len(self._pitches):
            return False
        return self._pitches[idx] == value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mode):
            return False
        return self._pitches.shape == other._pitches.shape and np.all(
            self._pitches == other._pitches
        )

    def __and__(self, other: Self) -> Self:
        newPitches = np.intersect1d(self._pitches, other._pitches)
        return self._newFromTrustedArray(newPitches)

    def __or__(self, other: Self) -> Self:
        return self.__class__(it.chain(self._pitches[1:], other._pitches[1:]))

    def combine(self, other: Self, offset: OPitch = OPitch()) -> Self:
        """
        Combine the current scale with another scale shifted by an interval. The resulting scale
        contains all the pitches of the current scale and the second scale's notes shifted by
        the given interval, repeating notes removed and sorted in ascending order.
        """
        return self.__class__(it.chain(self._pitches[1:], other._pitches + offset))

    def selfCombine(self, offset: OPitch = OPitch()) -> Self:
        """
        Similar to `combine`, but the second scale is the current scale itself.
        """
        return self.combine(self, offset)

    def __neg__(self) -> Self:
        newPitches = _scaleInvert(self._pitches)
        return self._newFromTrustedArray(newPitches)

    def __str__(self) -> str:
        return f"({', '.join(map(str, self._pitches))})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{str(self)}"

    def __hash__(self) -> int:
        return hash(tuple(self._pitches))


class _ModeCyclicAccessor:
    """Helper type providing cyclic indexing and slicing for `Mode` objects."""

    def __new__(cls, parent: Mode):
        return cls._newHelper(parent)

    @classmethod
    @lru_cache
    def _newHelper(cls, parent: Mode) -> Self:
        self = super().__new__(cls)
        self._parent = parent
        return self

    @overload
    def __getitem__(self, key: int) -> OPitch: ...

    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Mode: ...

    def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Mode:
        if isinstance(key, slice):
            if key.step == 0:
                if (
                    key.start is not None
                    and key.stop is not None
                    and key.start >= key.stop
                ):
                    raise IndexError("empty slice cannot make a scale")
                return Mode()
            negStep = key.step is not None and key.step < 0
            if negStep:
                roll = -key.start - 1 if key.stop is not None else -1
                key = slice(-1, key.stop, key.step)
            else:
                roll = -key.start if key.start is not None else 0
                key = slice(0, key.stop, key.step)
            newPitches = np.roll(self._parent._pitches, roll)[key]
            if len(newPitches) == 0:
                raise IndexError("empty slice cannot make a scale")
            newPitches -= newPitches[0]
            if negStep:
                newPitches[1:] *= -1
            return Mode._newFromTrustedArray(newPitches)
        elif isinstance(key, Iterable):
            key = np.array(list(set(key)))
            start = key[0]
            key -= start
            key %= len(self._parent)
            key.sort()
            newPitches = np.roll(self._parent._pitches, -start)[key]
            return Mode._newFromTrustedArray(newPitches)
        else:
            key %= len(self._parent)
            return self._parent._pitches[key]


class Modes:
    """Common scales in western music."""

    MAJOR = IONIAN = Mode(range(1, 7))
    HARMONIC_MAJOR = MAJOR.alter(5, -1)
    DORIAN = MAJOR.cyc[1:]
    PHRYGIAN = MAJOR.cyc[2:]
    LYDIAN = MAJOR.cyc[3:]
    MIXOLYDIAN = MAJOR.cyc[4:]
    MINOR = AOLIAN = MAJOR.cyc[5:]
    HARMONIC_MINOR = MINOR.alter(6, 1)
    MELODIC_MINOR = HARMONIC_MINOR.alter(5, 1)
    LOCRIAN = MAJOR.cyc[6:]
    MAJOR_PENTATONIC = CN_GONG = Mode(1, 2, 4, 5)
    CN_SHANG = MAJOR_PENTATONIC.cyc[1:]
    CN_JUE = MAJOR_PENTATONIC.cyc[2:]
    CN_ZHI = MAJOR_PENTATONIC.cyc[3:]
    MINOR_PENTATONIC = CN_YU = MAJOR_PENTATONIC.cyc[4:]
    WHOLE_TONE = WHOLE_TONE_SHARP = Mode(1, 2, OPitch(3, 1), OPitch(4, 1), OPitch(5, 1))
    WHOLE_TONE_FLAT = Mode(1, 2, OPitch(4, -1), OPitch(5, -1), OPitch(6, -1))
    BLUES = Mode(OPitch(2, -1), 3, OPitch(3, 1), 4, OPitch(6, -1))


class Scale(Sequence[OPitch], Set[OPitch]):
    def __init__(self, tonic: OPitch = OPitch(), mode: Mode = Modes.MAJOR):
        self._tonic = tonic
        self._mode = mode
        super().__init__()

    @property
    def tonic(self) -> OPitch:
        return self._tonic

    @property
    def mode(self) -> Mode:
        return self._mode

    def __len__(self):
        return len(self.mode)

    @overload
    def __getitem__(self, key: int) -> OPitch: ...

    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Self: ...

    def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Self:
        if isinstance(key, slice) or isinstance(key, Iterable):
            raise NotImplementedError
            # TODO: implement slicing and multiple indexing
        else:
            return self.tonic + self.mode._pitches[key]

    def __iter__(self) -> Iterator[OPitch]:
        for interval in self.mode:
            yield self.tonic + interval


OInterval = OPitch
Interval = Pitch
OIntervalSet = Mode
