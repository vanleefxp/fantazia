from __future__ import annotations

from typing import Any, Self
import typing as t
from collections.abc import Callable
from numbers import Real


from . import diatonic as _abc_diatonic
from ...utils.cls import cachedClassProp, cachedGetter

if t.TYPE_CHECKING:
    from .. import edo12


class Notation[OPType: "OPitch", PType: "Pitch"](_abc_diatonic.Notation[OPType, PType]):
    """Helper type for wrapping a `PitchBase` with a different type."""

    @property
    def acci(self) -> Real:
        return self.opitch.acci

    def __add__(self, other: Any) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._newHelper(self._p + other._p)

    def __neg__(self) -> Self:
        return self._newHelper(-self._p)

    def __eq__(self, value) -> bool:
        return isinstance(value, self.__class__) and self._p == value._p

    @cachedGetter
    def __hash__(self) -> int:
        return hash((self.__class__, self._p))

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[Any, ...]]:
        return (self._newHelper, (self._p,))


class OPitch[PType: "Pitch"](Notation[Self, PType], _abc_diatonic.OPitch[PType]):
    _p: edo12.OPitch

    @classmethod
    def _newImpl(cls, p: edo12.OPitch) -> Self:
        self = super().__new__(cls)
        self._p = p
        return self

    @cachedClassProp(key="_zero")
    def ZERO(cls) -> Self:
        from .. import edo12
        # import when needed to avoid circular import

        return cls._newHelper(edo12.OPitch.ZERO)

    @classmethod
    def _newImpl(cls, p: edo12.OPitch) -> Self:
        self = super().__new__(cls)
        self._p = p
        return self

    @classmethod
    def _fromStepAndAcci(cls, step: int, acci: Real) -> Self:
        from .. import edo12
        # import when needed to avoid circular import

        return cls._newHelper(edo12.OPitch._fromStepAndAcci(step, acci))

    @property
    def step(self) -> int:
        return self._p.step

    @property
    def acci(self) -> int:
        return self._p.acci


class Pitch[OPType: OPitch](Notation[OPType, Self], _abc_diatonic.Pitch[OPType]):
    @cachedClassProp(key="_zero")
    def ZERO(cls) -> Self:
        return cls._newHelper(cls.OPitch.ZERO)
