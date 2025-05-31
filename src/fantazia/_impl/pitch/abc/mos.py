from __future__ import annotations

from abc import ABCMeta, abstractmethod
import typing as t
from typing import Self, Any, overload
from collections.abc import Iterable, Iterator, Sequence
from numbers import Integral
import math
from functools import lru_cache
import warnings
import re

import numpy as np

from .base import PitchNotationBase, OPitchNotation, PitchNotation
from ...utils.bitmap import frozenbmap
from ...utils.cls import NewHelperMixin, cachedProp, cachedGetter, classProp

__all__ = [
    "MOSPatternBase",
    "MOSPattern",
    "MOSMode",
    "QuasiDiatonicMOSNotation",
    "OEDOMOSPitch",
    "EDOMOSPitch",
]

_MISSING = object()
_MOS_PATTERN_LS_RE = re.compile(r"^(?P<l_count>\d+)L(?P<s_count>\d+)s$", re.I)
_MOS_PATTERN_SL_RE = re.compile(r"^(?P<s_count>\d+)s(?P<l_count>\d+)L$", re.I)


@lru_cache
def _calcMosPattern(l_count: int, s_count: int) -> frozenbmap:
    """
    Compute the MOS pattern for a given number of large and small steps using [Bresenham's line
    algorithm](https://en.wikipedia.org/wiki/Bresenham's_line_algorithm).
    """
    steps = l_count + s_count
    pos = np.arange(steps) * l_count // steps
    diff = np.diff(pos, append=l_count)
    pattern = frozenbmap(diff.astype(bool), bufferLike=False)
    return pattern


@lru_cache
def _brightness2Rotate(l_count: int, s_count: int, brightness: int) -> int:
    length = l_count + s_count
    if brightness % length == 0:
        return 0

    from egcd import egcd
    # determine rotation by computing the inverse of `l_count` modulus `length`
    # using bezout's identity and extended Euclidean algorithm

    _, u, v = egcd(l_count, s_count)
    return (v - u) * brightness % length


class MOSPatternBase(metaclass=ABCMeta):
    __slots__ = ()

    @property
    @abstractmethod
    def pattern(self) -> MOSPattern:
        raise NotImplementedError

    @property
    def l_count(self) -> int:
        return self.pattern.l_count

    @property
    def s_count(self) -> int:
        return self.pattern.s_count

    def __len__(self) -> int:
        return self.l_count + self.s_count

    @property
    def steps(self) -> frozenbmap:
        return self.pattern.steps


class MOSPattern(MOSPatternBase, NewHelperMixin):
    """
    An abstract MOS pattern, which contains the number of large and small steps.
    """

    __slots__ = ("_l_count", "_s_count", "_rotate", "_steps", "_hash")

    _l_count: int
    _s_count: int

    if t.TYPE_CHECKING:  # pragma: no cover

        @overload
        def __new__(cls, src: str) -> Self: ...

        @overload
        def __new__(cls, l_count: Integral, s_count: Integral) -> Self: ...

    def __new__(cls, arg1=_MISSING, arg2=_MISSING) -> Self:
        if arg2 is _MISSING:
            if isinstance(arg1, cls):
                return arg1

            m = _MOS_PATTERN_LS_RE.match(arg1) or _MOS_PATTERN_SL_RE.match(arg1)
            if m is not None:
                l_count, s_count = int(m.group("l_count")), int(m.group("s_count"))
                gcd = math.gcd(l_count, s_count)
                l_count //= gcd
                s_count //= gcd
                return cls._newHelper(l_count, s_count)
            else:
                raise ValueError(f"Invalid MOS pattern: {arg1}")

        l_count, s_count = arg1, arg2
        if l_count <= 0 or s_count <= 0:
            raise ValueError("`l_count` and `s_count` must be positive integers")
        gcd = math.gcd(l_count, s_count)
        l_count = int(l_count // gcd)
        s_count = int(s_count // gcd)
        return cls._newHelper(l_count, s_count)

    @classmethod
    def _newImpl(cls, l_count: int, s_count: int) -> Self:
        self = super().__new__(cls)
        self._l_count = l_count
        self._s_count = s_count
        return self

    @property
    def pattern(self) -> MOSPattern:
        return self

    @property
    def l_count(self) -> int:
        return self._l_count

    @property
    def s_count(self) -> int:
        return self._s_count

    def __getitem__(
        self, idx: Integral | slice | Iterable[Integral]
    ) -> bool | frozenbmap:
        return self.steps[idx]

    def __iter__(self) -> Iterator[bool]:
        return iter(self.steps)

    def __reversed__(self) -> Iterator[bool]:
        return reversed(self.steps)

    @cachedProp
    def steps(self) -> frozenbmap:
        return _calcMosPattern(self.l_count, self.s_count)

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, MOSPattern):
            return False
        return (
            self._l_count == value._l_count
            and self._s_count == value._s_count
            and self._rotate == value._rotate
        )

    @cachedGetter
    def __hash__(self) -> int:
        return hash((self._l_count, self._s_count))

    def __str__(self) -> str:
        return "".join(map(lambda x: "L" if x else "s", self.steps))


def _resolveRotate(pattern, rotate, brightness) -> int:
    if brightness is not _MISSING:
        if rotate is not _MISSING:
            warnings.warn(
                "Both `brightness` and `rotate` are specified, `rotate` will be ignored"
            )
        rotate = _brightness2Rotate(pattern.l_count, pattern.s_count, brightness)
    elif rotate is _MISSING:
        rotate = pattern.l_count + pattern.s_count - 1
    else:
        rotate = int(rotate % len(pattern))
    return rotate


class MOSMode(MOSPatternBase, NewHelperMixin):
    __slots__ = ("_pattern", "_rotate", "_hash", "_steps")

    if t.TYPE_CHECKING:  # pragma: no cover

        @overload
        def __new__(cls, pattern: MOSPattern | str, /, *, rotate: Integral) -> Self: ...

        @overload
        def __new__(
            cls, l_count: Integral, s_count: Integral, /, *, rotate: Integral
        ) -> Self: ...

    _pattern: MOSPattern
    _rotate: int

    def __new__(
        self,
        arg0=_MISSING,
        arg1=_MISSING,
        /,
        *,
        rotate=_MISSING,
        brightness=_MISSING,
        **kwargs,
    ) -> Self:
        pattern = MOSPattern(arg0, arg1)
        rotate = _resolveRotate(pattern, rotate, brightness)
        return self._newImpl(pattern, rotate)

    @classmethod
    def _newImpl(cls, pattern, rotate) -> Self:
        self = super().__new__(cls)
        self._pattern = pattern
        self._rotate = rotate
        return self

    @property
    def pattern(self) -> MOSPattern:
        return self._pattern

    @property
    def rotate(self) -> int:
        return self._rotate

    @property
    def brightness(self) -> int:
        return (-self._rotate) * self.l_count % len(self)

    @cachedProp
    def steps(self) -> frozenbmap:
        return self.pattern.steps.roll(self.rotate)

    @cachedProp
    def diatonic(self) -> Sequence[int]:
        arr = np.empty(len(self.pattern), dtype=bool)
        arr[1:] = self.steps[1:]
        arr[0] = False
        return np.cumsum(arr)

    def __len__(self) -> int:
        return len(self.pattern)

    def __iter__(self) -> Iterator[int]:
        return map(int, self.diatonic)

    def __lshift__(self, shift: Integral) -> Self:
        return self._newImpl(self.pattern, (self.rotate - shift) % len(self))

    def __rshift__(self, shift: Integral) -> Self:
        return self._newImpl(self.pattern, (self.rotate + shift) % len(self))

    def __eq__(self, other: Any):
        if not isinstance(other, self.__class__):
            return False
        return self.pattern == other.pattern and self.rotate == other.rotate

    def __hash__(self) -> int:
        return hash((self.pattern, self.rotate))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.l_count}, {self.s_count}, {self.rotate})"
        )

    def __str__(self) -> str:
        return "".join(map(lambda x: "L" if x else "s", self.steps))


Sequence.register(MOSPattern)
Sequence.register(MOSMode)


class IntMOSPattern(MOSPatternBase, NewHelperMixin):
    """
    MOS pattern with integer large and small step sizes.
    """

    __slots__ = ("_pattern", "_l_size", "_s_size", "_hash")

    _pattern: MOSPattern
    _l_size: int
    _s_size: int

    @classmethod
    def _fromEdoAndNSteps(cls, edo: int, n_steps: int) -> Self:
        gcd = math.gcd(edo, n_steps)
        edo //= gcd
        n_steps //= gcd

        s_size = edo // n_steps
        l_size = s_size + 1
        l_count = edo - s_size * n_steps
        s_count = n_steps - l_count

        pattern = MOSPattern(l_count, s_count)
        return cls._newHelper(pattern, l_size, s_size)

    @classmethod
    def _newImpl(cls, pattern, l_size, s_size) -> Self:
        self = super().__new__(cls)
        self._pattern = pattern
        self._l_size = l_size
        self._s_size = s_size
        return self

    @property
    def pattern(self) -> MOSPattern:
        return self._pattern

    @property
    def l_size(self) -> int:
        return self._l_size

    @property
    def s_size(self) -> int:
        return self._s_size

    @property
    def edo(self) -> int:
        """
        Number of equal divisions of the octave the current MOS pattern provides.
        """
        return (self.l_size * self.l_count + self.s_size * self.s_count) * self.n_loops

    @property
    def sharpness(self) -> int:
        """
        Number of EDO steps a sharp sign raises (or a flat sign lowers).
        Equals the difference between large and small step sizes.
        """
        return self.l_size - self.s_size

    def __getitem__(
        self, idx: Integral | slice | Iterable[Integral]
    ) -> int | tuple[int, ...]:
        if isinstance(idx, (slice, Iterable)):
            return tuple(
                map(
                    lambda isLarge: self.l_size if isLarge else self.s_size,
                    self.pattern[idx],
                )
            )
        else:
            return self.l_size if self.pattern[idx] else self.s_size

    def __iter__(self) -> Iterator[int]:
        return map(
            lambda isLarge: self.l_size if isLarge else self.s_size,
            self.pattern,
        )

    def __reversed__(self) -> Iterator[int]:
        return map(
            lambda isLarge: self.s_size if isLarge else self.l_size,
            reversed(self.pattern),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern}, L={self.l_size}, s={self.s_size})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            self.pattern == other.pattern
            and self.l_size == other.l_size
            and self.s_size == other.s_size
        )

    @cachedGetter
    def __hash__(self) -> int:
        return hash((self.pattern, self.l_size, self.s_size))


class QuasiDiatonicMOSNotation[OPType: "OEDOMOSPitch", PType: "EDOMOSPitch"](
    PitchNotationBase[OPType, PType]
):
    @classProp
    @abstractmethod
    def pattern(cls) -> IntMOSPattern:
        raise NotImplementedError

    @classProp
    def edo(cls) -> int:
        return cls.pattern.edo

    @classProp
    def diatonic(cls) -> Sequence[int]:
        l_size, s_size, length = (
            cls.pattern.l_size,
            cls.pattern.s_size,
            len(cls.pattern),
        )
        return (
            np.cumsum(cls.pattern.steps) * (l_size - s_size)
            + np.arange(length) * s_size
        )

    @property
    def step(self) -> int:
        raise NotImplementedError

    @property
    def acci(self) -> int:
        raise NotImplementedError


class OEDOMOSPitch[PType: "EDOMOSPitch"](
    OPitchNotation[PType], QuasiDiatonicMOSNotation[Self, PType]
): ...


class EDOMOSPitch[OPType: OEDOMOSPitch](
    PitchNotation[OPType], QuasiDiatonicMOSNotation[OPType, Self]
): ...
