# from __future__ import annotations

# import itertools as it
# import typing as t
# from abc import ABCMeta
# from bisect import bisect_left
# from collections.abc import Iterable, Iterator, Sequence, Set
# from functools import lru_cache
# from numbers import Integral, Real
# from typing import Self, overload, Any

# import numpy as np
# from sortedcontainers import SortedSet

# from .utils.cls import cachedGetter

# # TODO)) put utils in a separate library

# __all__ = [
#     "ostep",
#     "step",
#     "acci",
#     "AcciPref",
#     "DegMaps",
#     "AcciPrefs",
#     "Modes",
#     "DegMap",
#     "edo12",
#     "OPitch",
#     "oP",
#     "Pitch",
#     "P",
#     "Mode",
#     "Scale",
#     "STEPS_CO5",
#     "MAJOR_SCALE_TONES_CO5",
#     "MAJOR_SCALE_TONES",
#     "MAJOR_SCALE_TONES_SET",
#     "PERFECTABLE_STEPS",
#     "STEP_NAMES",
#     "STEP_NAMES_SET",
# ]


# class DegMap(Sequence[Real]):
#     """A mapping from scale degrees to tones in an octave."""

#     __slots__ = ("_tones", "_pitches", "_mode")

#     if t.TYPE_CHECKING:  # pragma: no cover

#         @overload
#         def __new__(cls, tones: Iterable[Real]): ...

#         @overload
#         def __new__(cls, *tones: Real): ...

#         @overload
#         def alter(self, idx: int, acci: Real) -> Self: ...

#         @overload
#         def alter(self, acci: Iterable[Real]) -> Self: ...

#         @overload
#         def alter(self, idx: Iterable[int], acci: Real | Iterable[Real]) -> Self: ...

#     def __new__(cls, *args) -> Self:
#         if len(args) == 1 and isinstance(args[0], Iterable):
#             tones = np.array(tuple(args[0]))
#         else:
#             tones = np.array(args)
#         if len(tones) != 7:
#             raise ValueError(f"Expected 7 tones, got {len(tones)}")

#         # The first element is always 0. Omitting it can save space, but will bring
#         # more trouble working with the array later on.
#         # So here I choose not to omit it.

#         tones -= tones[0]
#         return cls._newHelper(tones)

#     @classmethod
#     def _newHelper(cls, tones: np.ndarray):
#         # caching is not enabled because `numpy` array is not hashable
#         self = super().__new__(cls)
#         tones.flags.writeable = False
#         self._tones = tones
#         return self

#     @property
#     def p(self) -> Sequence[OPitch]:
#         """Access degree map elements as pitch objects."""
#         if not hasattr(self, "_pitches"):
#             self._pitches = _DegMapPitchesView(self)
#         return self._pitches

#     @property
#     def mode(self) -> Mode:
#         """Turn the degree map into a mode."""
#         if not hasattr(self, "_mode"):
#             self._mode = Mode(self.p)
#         return self._mode

#     def __len__(self):
#         return len(self._tones)

#     def __getitem__(
#         self, key: int | str | OPitch | slice | Iterable[int]
#     ) -> int | Sequence[int]:
#         if isinstance(key, tuple):
#             key = np.array(key)
#         if isinstance(key, Integral):
#             return self._tones[key]
#         else:
#             key = OPitch(key)
#             return self._tones[key.deg] + key.acci

#     def __iter__(self):
#         return iter(self._tones)

#     def __reversed__(self):
#         return reversed(self._tones)

#     def __hash__(self):
#         return hash(self._tones.tobytes())

#     def __eq__(self, other: Any):
#         if not isinstance(other, DegMap):
#             return False
#         return np.all(self._tones == other._tones)

#     def __add__(self, other: Iterable[Real]) -> Self:
#         if not isinstance(other, np.ndarray):
#             if not isinstance(other, Sequence):
#                 other = tuple(other)
#             other = np.array(other)
#         newTones = self._tones + other
#         if newTones[0] != 0:
#             newTones -= newTones[0]
#         return self._newHelper(newTones)

#     def alter(
#         self,
#         arg1: int | Iterable[int] | Iterable[Real],
#         arg2: Real | Iterable[Real] | None = None,
#     ) -> Self:
#         if arg2 is None:
#             if isinstance(arg1, Iterable):
#                 return self + arg1
#             else:
#                 return self
#         if not isinstance(arg1, Iterable):
#             arg1 = (arg1,)
#         if not isinstance(arg2, Iterable):
#             arg2 = it.repeat(arg2)
#         # store alteration amounts in a list to avoid `numpy` array type issues
#         lst = [0 for _ in range(7)]
#         for idx, acci in zip(arg1, arg2):
#             lst[idx % 7] += acci
#         return self + lst

#     def roll(self, shift: int) -> Self:
#         newTones = np.roll(self._tones, shift)
#         newTones -= newTones[0]
#         newTones %= 12
#         return self._newHelper(newTones)

#     def diff(self) -> np.ndarray[int]:
#         return np.diff(self._tones, append=12)

#     def __repr__(self):
#         return f"{self.__class__.__name__}({self._tones})"


# class DegMaps:
#     MAJOR = IONIAN = DegMap(MAJOR_SCALE_TONES)
#     HARMONIC_MAJOR = MAJOR.alter(5, -1)
#     DORIAN = MAJOR.roll(-1)
#     PHRYGIAN = MAJOR.roll(-2)
#     LYDIAN = MAJOR.roll(1)
#     MIXOLYDIAN = MAJOR.roll(2)
#     MINOR = AEOLIAN = MAJOR.roll(3)
#     HARMONIC_MINOR = MINOR.alter(6, 1)
#     MELODIC_MINOR = HARMONIC_MINOR.alter(5, 1)
#     LOCRIAN = MAJOR.roll(4)


# class _DegMapPitchesView(Sequence["OPitch"], Set["OPitch"]):
#     __slots__ = ("_parent",)

#     def __init__(self, parent: DegMap):
#         self._parent = parent

#     def __len__(self):
#         return 7

#     def __iter__(self) -> Iterator["OPitch"]:
#         return map(OPitch._fromStepAndTone, range(7), self._parent._tones)

#     def __contains__(self, value: Any):
#         if not isinstance(value, edo12):
#             return False
#         return self._parent._tones[value.step % 7] == value.tone

#     def index(self, value: edo12) -> OPitch:
#         """Find the degree of pitch in the degree map."""
#         value = value.opitch
#         step = value.step
#         acci = value.tone - self._parent._tones[step]
#         return OPitch._newHelper(step, acci)

#     def __eq__(self, other) -> bool:
#         return self._parent == other._parent

#     def __getitem__(self, key: int) -> int:
#         if isinstance(key, slice):
#             return self._slice(key)
#         if isinstance(key, Iterable) and not isinstance(key, str):
#             return self._multiIndex(key)
#         return self._getItem(key)

#     def _getItem(self, key: int | str | OPitch) -> OPitch:
#         if isinstance(key, Integral):
#             return OPitch._fromStepAndTone(key, self._parent._tones[key])
#         if not isinstance(key, OPitch):
#             key = OPitch(key)
#         return OPitch._fromStepAndTone(
#             key.step, self._parent._tones[key.step] + key.acci
#         )

#     def _slice(self, key: slice) -> Sequence[OPitch]:
#         steps = np.arange(7)[key]
#         tones = self._parent._tones[steps]
#         return np.array(
#             [OPitch._fromStepAndTone(s, t) for s, t in zip(steps, tones)], dtype=object
#         )

#     def _multiIndex(self, key: Iterable[int | str | OPitch]) -> Sequence[OPitch]:
#         return np.array([self._getItem(k) for k in key], dtype=object)


# # Diatonic Notation System


# # EDO notation system by chain-of-fifths method


# def _modeAlter(
#     pitches: np.ndarray[OPitch], step: int, acci: Real
# ) -> np.ndarray[OPitch]:
#     if acci != 0:
#         if step == 0:
#             pitches[1:] = np.array([p.alter(-acci) for p in pitches[1:]])
#         else:
#             pitches[step] = pitches[step].alter(acci)
#     return pitches


# def _invertPitches(pitches: np.ndarray[OPitch]) -> np.ndarray[OPitch]:
#     pitches = -pitches
#     pitches[1:] = pitches[1:][::-1]
#     return pitches


# class Mode(Sequence[OPitch], Set[OPitch], metaclass=ABCMeta):
#     """
#     A **mode** is a sequence of unique octave intervals in ascending order, starting from
#     perfect unison.
#     """

#     __slots__ = ("_pitches", "_cyc", "_hash")

#     if t.TYPE_CHECKING:  # pragma: no cover

#         @overload
#         def __new__(cls, pitches: Iterable[OPitch | int | str]) -> Self: ...

#         @overload
#         def __new__(cls, *pitches: OPitch | int | str) -> Self: ...

#         @overload
#         def __getitem__(self, key: int) -> OPitch: ...

#         @overload
#         def __getitem__(self, key: slice | Iterable[int]) -> Self:
#             """
#             Extract a new scale from part of the current scale.
#             """
#             ...

#         @overload
#         def alter(self, idx: int, acci: Real) -> Self: ...

#         @overload
#         def alter(self, idx: Iterable[int], acci: Iterable[Real] | Real) -> Self: ...

#         @overload
#         def alter(self, acci: Iterable[Real]) -> Self: ...

#     def __new__(cls, *args) -> Self:
#         if (
#             len(args) == 1
#             and isinstance(args[0], Iterable)
#             and not isinstance(args[0], str)
#         ):
#             pitches = args[0]
#         else:
#             pitches = args
#         pitches = np.array([OPitch(p) for p in pitches])
#         if pitches[0] != OPitch.ZERO:
#             pitches -= pitches[0]
#         pitches[1:].sort()
#         return cls._newHelper(pitches)

#     @classmethod
#     def _newHelper(cls, pitches: np.ndarray[OPitch]) -> Self:
#         self = object.__new__(cls)
#         self._pitches = pitches
#         self._pitches.flags.writeable = False
#         return self

#     def __len__(self) -> int:
#         return len(self.pitches)

#     def __contains__(self, value: Any) -> bool:
#         # a scale is an ordered sequence
#         # so use binary search
#         if not isinstance(value, edo12):
#             return False
#         idx = bisect_left(self, value)
#         return idx < len(self) and self[idx] == value

#     @property
#     def pitches(self) -> Sequence[OPitch]:
#         return self._pitches

#     @property
#     def cyc(self) -> _ModeCyclicAccessor:
#         """Cyclic slicing and access support."""
#         return _ModeCyclicAccessor(self)

#     def diff(self) -> Iterable[OPitch]:
#         """
#         Returns the interval structure of the scale, i.e., the differences between adjacent
#         pitches.
#         """
#         return np.diff(self.pitches, append=OPitch.ZERO)

#     def __str__(self) -> str:
#         return f"({', '.join(map(str, self.pitches))})"

#     def __repr__(self) -> str:
#         return f"Mode{str(self)}"

#     def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Self:
#         if isinstance(key, slice):  # generate a new scale by slicing
#             return self._slice(self, key)[0]
#         elif isinstance(key, Iterable):  # generate a new scale by a set of indices
#             return self._multiIndex(self, key)[0]
#         else:  # get a pitch by index
#             return self._getItem(key)

#     def _getItem(self, key: int) -> OPitch:
#         # getting a single item
#         return self.pitches[key]

#     def _slice(self, key: slice) -> tuple[Mode, OPitch]:
#         # create a new mode from slicing
#         start, _, step = key.indices(len(self))
#         newPitches = self.pitches[key].copy()
#         if len(newPitches) == 0:
#             raise IndexError("empty slice cannot make a scale")
#         if step < 0:
#             newPitches = np.roll(newPitches, 1)
#             startPitch = newPitches[0]
#             newPitches -= startPitch
#             newPitches[1:] *= -1
#         else:
#             if start > 0:  # not starting from first note
#                 startPitch = newPitches[0]
#                 newPitches -= startPitch
#             else:
#                 startPitch = OPitch.ZERO
#         return Mode._newHelper(newPitches), startPitch

#     def _cycSlice(self, key: slice) -> tuple[Mode, OPitch]:
#         # create a new mode from cyclic slicing
#         if key.step == 0:
#             if key.start is not None and key.stop is not None and key.start >= key.stop:
#                 raise IndexError("empty slice cannot make a scale")
#             return Mode(), OPitch.ZERO
#         negStep = key.step is not None and key.step < 0
#         if negStep:
#             roll = -key.start - 1 if key.start is not None else -1
#             key = slice(-1, key.stop, key.step)
#         else:
#             roll = -key.start if key.start is not None else 0
#             key = slice(0, key.stop, key.step)
#         newPitches = np.roll(self.pitches, roll)[key].copy()
#         if len(newPitches) == 0:
#             raise IndexError("empty slice cannot make a scale")
#         startPitch = newPitches[0]
#         newPitches -= startPitch
#         if negStep:
#             newPitches[1:] = -newPitches[1:]
#         return Mode._newHelper(newPitches), startPitch

#     def _multiIndex(self, key: Iterable[int]) -> tuple[Mode, OPitch]:
#         # create a new mode from a set of indices
#         indices = SortedSet(key)
#         if len(indices) == 0:
#             raise IndexError("empty set cannot make a scale")
#         newPitches = self.pitches[list(indices)].copy()
#         if indices[0] > 0:
#             startPitch = newPitches[0]
#             newPitches -= startPitch
#         else:
#             startPitch = OPitch.ZERO
#         return Mode._newHelper(newPitches), startPitch

#     def _cycMultiIndex(self, key: Iterable[int]) -> tuple[Mode, OPitch]:
#         # create a new mode from a set of indices in cyclic order
#         # the first index provided is regarded as the new tonic
#         key = np.array(list(set(key)))
#         start = key[0]
#         key -= start
#         key %= len(self)
#         key.sort()
#         newPitches = np.roll(self.pitches, -start)[key]
#         startPitch = newPitches[0]
#         newPitches -= startPitch
#         return Mode._newHelper(newPitches), startPitch

#     def alter(
#         self,
#         arg1: int | Iterable[int] | Iterable[Real],
#         arg2: Real | Iterable[Real] | None = None,
#     ) -> Self:
#         """
#         Apply alterations to the scale by adjusting the accidentals of specific pitches.
#         """
#         if arg2 is None:
#             if isinstance(arg1, Iterable):
#                 newPitches = self._pitches.copy()
#                 for i, acci in enumerate(arg1):
#                     _modeAlter(newPitches, i, acci)
#             else:
#                 return self
#         else:
#             if isinstance(arg1, Iterable):
#                 if isinstance(arg2, Iterable):
#                     newPitches = self._pitches.copy()
#                     for i, acci in zip(arg1, arg2):
#                         _modeAlter(newPitches, i, acci)
#                 else:
#                     newPitches = self._pitches.copy()
#                     for i in arg1:
#                         _modeAlter(newPitches, i, arg2)
#             else:
#                 newPitches = self._pitches.copy()
#                 _modeAlter(newPitches, arg1, arg2)
#         return Mode(newPitches)

#     def combine(self, other: Self, offset: OPitch = OPitch.ZERO) -> Mode:
#         """
#         Combine the current scale with another scale shifted by an interval. The resulting scale
#         contains all the pitches of the current scale and the second scale's notes shifted by
#         the given interval, repeating notes removed and sorted in ascending order.
#         """
#         return Mode(it.chain(self._pitches[1:], other._pitches + offset))

#     def stack(self, offset: OPitch = OPitch.ZERO) -> Mode:
#         """
#         Similar to `combine`, but the second scale is the current scale itself.
#         """
#         return self.combine(self, offset)

#     def __and__(self, other: Self) -> Mode:
#         newPitches = np.intersect1d(self.pitches, other.pitches)
#         return Mode._newHelper(newPitches)

#     def __or__(self, other: Self) -> Mode:
#         return Mode(it.chain(self.pitches[1:], other.pitches[1:]))

#     def __neg__(self) -> Self:
#         newPitches = _invertPitches(self._pitches)
#         return Mode._newHelper(newPitches)

#     def __iter__(self) -> Iterator[OPitch]:
#         return iter(self.pitches)

#     def __reversed__(self) -> Iterator[OPitch]:
#         return reversed(self.pitches)

#     @cachedGetter
#     def __hash__(self) -> int:
#         return hash(tuple(self.pitches))

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, Mode):
#             return False
#         return self._pitches.shape == other._pitches.shape and np.all(
#             self._pitches == other._pitches
#         )


# class _ModeCyclicAccessor:
#     """Helper type providing cyclic indexing and slicing for `Mode` objects."""

#     __slots__ = ("_parent",)

#     if t.TYPE_CHECKING:  # pragma: no cover

#         @overload
#         def __getitem__(self, key: int) -> OPitch: ...

#         @overload
#         def __getitem__(self, key: slice | Iterable[int]) -> Mode: ...

#     def __new__(cls, parent: Mode):
#         return cls._newHelper(parent)

#     @classmethod
#     @lru_cache
#     def _newHelper(cls, parent: Mode) -> Self:
#         self = super().__new__(cls)
#         self._parent = parent
#         return self

#     def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Mode:
#         if isinstance(key, slice):
#             return self._parent._cycSlice(key)[0]
#         elif isinstance(key, Iterable):
#             return self._parent._cycMultiIndex(key)[0]
#         else:
#             key %= len(self._parent)
#             return self._parent._getItem(key)


# class Modes:
#     """Common modes in western music."""

#     MAJOR = IONIAN = Mode(range(7))
#     HARMONIC_MAJOR = MAJOR.alter(5, -1)
#     DORIAN = MAJOR.cyc[1:]
#     PHRYGIAN = MAJOR.cyc[2:]
#     LYDIAN = MAJOR.cyc[3:]
#     MIXOLYDIAN = MAJOR.cyc[4:]
#     MINOR = AOLIAN = MAJOR.cyc[5:]
#     HARMONIC_MINOR = MINOR.alter(6, 1)
#     MELODIC_MINOR = HARMONIC_MINOR.alter(5, 1)
#     LOCRIAN = MAJOR.cyc[6:]
#     MAJOR_PENTATONIC = CN_GONG = Mode(0, 1, 2, 4, 5)
#     CN_SHANG = MAJOR_PENTATONIC.cyc[1:]
#     CN_JUE = MAJOR_PENTATONIC.cyc[2:]
#     CN_ZHI = MAJOR_PENTATONIC.cyc[3:]
#     MINOR_PENTATONIC = CN_YU = MAJOR_PENTATONIC.cyc[4:]
#     WHOLE_TONE = WHOLE_TONE_SHARP = Mode(
#         0, 1, 2, OPitch(3, 1), OPitch(4, 1), OPitch(5, 1)
#     )
#     WHOLE_TONE_FLAT = Mode(0, 1, 2, OPitch(4, -1), OPitch(5, -1), OPitch(6, -1))
#     BLUES = Mode(0, OPitch(2, -1), 3, OPitch(3, 1), 4, OPitch(6, -1))


# class Scale(Sequence[OPitch], Set[OPitch]):
#     """A **scale** is a sequence of pitches in a specific mode, starting from a tonic."""

#     __slots__ = ("_tonic", "_mode", "_cyc", "_pitches")

#     if t.TYPE_CHECKING:  # pragma: no cover

#         @overload
#         def __getitem__(self, key: int) -> OPitch: ...

#         @overload
#         def __getitem__(self, key: slice | Iterable[int]) -> Self: ...

#     def __new__(
#         cls, tonic: OPitch | int | str = OPitch.ZERO, mode: Mode = Modes.MAJOR
#     ) -> Self:
#         if not isinstance(tonic, OPitch):
#             tonic = OPitch(tonic)
#         return cls._newHelper(tonic, mode)

#     @classmethod
#     @lru_cache
#     def _newHelper(cls, tonic: OPitch, mode: Mode) -> Self:
#         self = super().__new__(cls)
#         self._tonic = tonic
#         self._mode = mode
#         return self

#     @property
#     def tonic(self) -> OPitch:
#         return self._tonic

#     @property
#     def mode(self) -> Mode:
#         return self._mode

#     @property
#     def pitches(self) -> Sequence[OPitch]:
#         if not hasattr(self, "_pitches"):
#             self._pitches = self._mode.pitches + self._tonic
#             self._pitches.flags.writeable = False
#         return self._pitches

#     @property
#     def cyc(self) -> _ScaleCyclicAccessor:
#         """Cyclic slicing and access support."""
#         if not hasattr(self, "_cyc"):
#             self._cyc = _ScaleCyclicAccessor(self)
#         return self._cyc

#     def diff(self) -> Sequence[OPitch]:
#         """
#         Returns the interval structure of the scale, i.e., the differences between adjacent
#         pitches.
#         """
#         return self.mode.diff()

#     def __len__(self):
#         return len(self.mode)

#     def __contains__(self, value: object):
#         if isinstance(value, OPitch):
#             return (value - self.tonic) in self.mode
#         return False

#     def __eq__(self, other: object):
#         if isinstance(other, Scale):
#             return self.tonic == other.tonic and self.mode == other.mode
#         return False

#     def __add__(self, other: OPitch) -> Self:
#         return self.__class__(self.tonic + other, self.mode)

#     def __sub__(self, other) -> Self:
#         return self.__class__(self.tonic - other, self.mode)

#     def __neg__(self) -> Self:
#         return self.__class__(self.tonic, -self.mode)

#     def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Self:
#         if isinstance(key, slice):
#             newMode, startPitch = self.mode._slice(key)
#             return self.__class__(self.tonic + startPitch, newMode)
#         elif isinstance(key, Iterable):
#             newMode, startPitch = self.mode._multiIndex(key)
#             return self.__class__(self.tonic + startPitch, newMode)
#         else:
#             return self.tonic + self.mode._getItem(key)

#     def __iter__(self) -> Iterator[OPitch]:
#         for interval in self.mode:
#             yield self.tonic + interval

#     def __repr__(self):
#         return f"{self.__class__.__name__}{str(self)}"

#     def __str__(self) -> str:
#         return f"({', '.join(map(str, self))})"

#     def __hash__(self):
#         return hash((self.tonic, self.mode))


# class _ScaleCyclicAccessor:
#     """Helper type providing cyclic indexing and slicing for `Scale` objects."""

#     __slots__ = ("_parent",)

#     if t.TYPE_CHECKING:  # pragma: no cover

#         @overload
#         def __getitem__(self, key: int) -> OPitch: ...

#         @overload
#         def __getitem__(self, key: slice | Iterable[int]) -> Scale: ...

#     def __new__(cls, parent: Scale):
#         return cls._newHelper(parent)

#     @classmethod
#     @lru_cache
#     def _newHelper(cls, parent: Scale) -> Self:
#         self = super().__new__(cls)
#         self._parent = parent
#         return self

#     def __getitem__(self, key: int | slice | Iterable[int]) -> OPitch | Scale:
#         if isinstance(key, slice):
#             newMode, startPitch = self._parent.mode._cycSlice(key)
#             return Scale._newHelper(self._parent.tonic + startPitch, newMode)
#         elif isinstance(key, Iterable):
#             newMode, startPitch = self._parent.mode._cycMultiIndex(key)
#             return Scale._newHelper(self._parent.tonic + startPitch, newMode)
#         else:
#             key %= len(self._parent.mode)
#             return self._parent[key]
