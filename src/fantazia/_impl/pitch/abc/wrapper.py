from __future__ import annotations

from typing import Any, Self
from collections.abc import Callable
from numbers import Real


from .diatonic import DiatonicPitchBase, ODiatonicPitch, DiatonicPitch
from ..edo12 import OPitch
from ...utils.cls import cachedClassProp, cachedGetter


class PitchWrapperBase[OPType: "OPitchWrapper", PType: "PitchWrapper"](
    DiatonicPitchBase[OPType, PType]
):
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


class OPitchWrapper[PType: "PitchWrapper"](
    PitchWrapperBase[Self, PType], ODiatonicPitch[PType]
):
    _p: OPitch

    @classmethod
    def _newImpl(cls, p: OPitch) -> Self:
        self = super().__new__(cls)
        self._p = p
        return self

    @cachedClassProp(key="_zero")
    def ZERO(cls) -> Self:
        return cls._newHelper(OPitch.ZERO)

    @classmethod
    def _newImpl(cls, p: OPitch) -> Self:
        self = super().__new__(cls)
        self._p = p
        return self

    @classmethod
    def _fromStepAndAcci(cls, step: int, acci: Real) -> Self:
        return cls._newHelper(OPitch._fromStepAndAcci(step, acci))

    @property
    def step(self) -> int:
        return self._p.step

    @property
    def acci(self) -> int:
        return self._p.acci


class PitchWrapper[OPType: OPitchWrapper](
    PitchWrapperBase[OPType, Self], DiatonicPitch[OPType]
):
    @cachedClassProp(key="_zero")
    def ZERO(cls) -> Self:
        return cls._newHelper(cls.OPitch.ZERO)
