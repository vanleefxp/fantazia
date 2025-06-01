from __future__ import annotations

from numbers import Real
from functools import lru_cache
from typing import Self
from collections.abc import Sequence

from bidict import bidict
import numpy as np
import pyrsistent as pyr

from . import quasiDiatonic as _abc_quasiDiatonic
from ...utils.cls import NewHelperMixin, classProp
from ...utils.number import RMode, rdivmod, resolveInt


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
"""
Major scale tones in increasing order. Equivalent to
`((np.arange(7) * 2 + 1) % 7 - 1) * 7 % 12`.

**Value**: `np.array([0, 2, 4, 5, 7, 9, 11])`
"""
MAJOR_SCALE_TONES.flags.writeable = False

MAJOR_SCALE_TONES_SET = frozenset(MAJOR_SCALE_TONES)
"""major scale tones as a set"""

PERFECTABLE_STEPS = frozenset((0, 3, 4))
"""Collection of interval step difference values that can have quality "perfect"."""

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

_solfegeNamesInvMap = pyr.pmap(
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


@lru_cache
def _qualMap(step: int, acci: Real) -> Real:
    if step % 7 in PERFECTABLE_STEPS:
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
    if step in PERFECTABLE_STEPS:
        return _prefectQualInvMap(qual)
    else:
        return _majorQualInvMap(qual)


def _qual2Str_int(qual: int, *, augDimThresh: int = 3) -> str:
    """Interval quality to string, but for integers only."""
    if qual > augDimThresh + 1:
        return f"[A*{qual - 1}]"
    elif qual > 1:
        return "A" * (qual - 1)
    elif qual < -augDimThresh - 1:
        return f"[d*{-qual - 1}]"
    elif qual < -1:
        return "d" * (-qual - 1)
    else:
        return _intervalQualityMap.inv[qual]


def _qual2Str(
    qual: Real, *, rmode: RMode | str = RMode.D, augDimThresh: int = 3
) -> str:
    """Interval quality to string."""
    q, r = rdivmod(qual, 1, rmode=rmode)
    if r == 0:
        return _qual2Str_int(q, augDimThresh=augDimThresh)
    else:
        return f"{_qual2Str_int(q)}[{round(r, 3):+}]"


def _acci2Str(acci: Real, *, symbolThresh: int = 3, maxDecimalDigits: int = 3) -> str:
    if acci == 0:
        return ""
    elif acci > 0:
        if acci.is_integer():
            acci = int(acci)
            if acci <= symbolThresh:
                return "+" * acci
            else:
                return f"[{acci:+d}]"
        else:
            return f"[{round(acci, maxDecimalDigits):+}]"
    else:
        if acci.is_integer():
            acci = int(acci)
            if acci >= -symbolThresh:
                return "-" * abs(acci)
            else:
                return f"[{acci:+d}]"
        else:
            return f"[{round(acci, maxDecimalDigits):+}]"


class Notation[OPType: "OPitch", PType: "Pitch"](
    _abc_quasiDiatonic.Notation[OPType, PType]
):
    __slots__ = ()

    @classProp
    def n_steps(self) -> int:
        return 7

    @property
    def qual(self) -> Real:
        """
        Interval quality when regarding the pitch as an interval.
        """
        return _qualMap(self.step, self.acci)

    def interval(self, *, rmode: RMode | str = RMode.D, **kwargs) -> str:
        "String representation of the pitch as an interval."
        return f"{_qual2Str(self.qual, rmode=rmode)}{self.step + 1}"

    def __repr__(self):
        return f'{self.__class__.__name__}("{self!s}")'


class OPitch[PType: "Pitch"](
    _abc_quasiDiatonic.OPitch[PType], Notation[Self, PType], NewHelperMixin
):
    __slots__ = ()

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
        return cls._fromStepAndAcci(step, acci)

    def __str__(self) -> str:
        return f"{STEP_NAMES[self.step]}{_acci2Str(self.acci)}"


class Pitch[OPType: OPitch](_abc_quasiDiatonic.Pitch[OPType], Notation[OPType, Self]):
    __slots__ = ()

    def interval(self, *, compound: bool = False, **kwargs):
        if self < self.ZERO:  # negative interval
            return f"-{(-self).interval(compound=compound, **kwargs)}"
        else:
            return super().interval(compound=compound, **kwargs)

    def __str__(self):
        return f"{self.opitch!s}_{self.o}"
