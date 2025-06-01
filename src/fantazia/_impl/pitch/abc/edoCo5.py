from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Self
import math

import numpy as np

from . import diatonic as _abc_diatonic, edo as _abc_edo
from ...utils.cls import cachedClassProp

_DIATONIC_REORDER = np.arange(1, 14, 2) % 7
_DIATONIC_REORDER.flags.writeable = False

__all__ = ["Notation", "OPitch", "Pitch"]


class Notation[OPType: "OPitch", PType: "Pitch"](
    _abc_diatonic.Notation[OPType, PType],
    _abc_edo.Notation[OPType, PType],
):
    """
    [**Chain-of-fifth notation**](https://en.xen.wiki/w/Chain-of-fifths_notation) for EDO
    tuning systems. This tuning system choses the best EDO approximation of a perfect
    fifth (P5) interval and generate diatonic pitches in chain-of-fifths order.
    """

    __slots__ = ()

    @cachedClassProp
    def fifthSize(cls) -> int:
        """
        Number of EDO steps corresponding to a perfect fifth (P5) interval.
        """
        return round(math.log2(1.5) * cls.edo)

    @cachedClassProp
    def sharpness(cls) -> int:
        """
        Number of EDO steps a sharp sign raises.

        **See**: <https://en.xen.wiki/w/Sharpness>
        """
        return cls.fifthSize * 7 - cls.edo * 4

    @cachedClassProp
    def diatonic(cls) -> Sequence[Integral]:
        """
        A sequence denoting mapping from diatonic steps to EDO steps.
        """
        res = np.arange(-1, 6) * cls.fifthSize - cls.edo * (np.arange(-1, 6) // 2)
        res = res[_DIATONIC_REORDER]
        res.flags.writeable = False
        return res


class OPitch[PType: "Pitch"](
    _abc_diatonic.OPitch[PType], _abc_edo.OPitch[PType], Notation[Self, PType]
):
    __match_args__ = ("step", "acci", "tone")
    __slots__ = ()


class Pitch[OPType: OPitch](
    _abc_diatonic.Pitch[OPType], _abc_edo.Pitch[OPType], Notation[OPType, Self]
):
    __match_args__ = ("opitch", "o", "step", "acci", "tone")
    __slots__ = ()
