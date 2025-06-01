from abc import abstractmethod
from typing import Self, Literal
from numbers import Real
from functools import lru_cache

from . import base as _abc_base
from ...utils.cls import classProp, NewHelperMixin


class Notation[OPType: "OPitch", PType: "Pitch"](_abc_base.Notation[OPType, PType]):
    __slots__ = ()

    @classProp
    @abstractmethod
    def n_steps(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def step(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def acci(self) -> Real:
        """An accidental value in semitones that modifies the pitch."""
        raise NotImplementedError


class OPitch[PType: "Pitch"](
    _abc_base.OPitch[PType], Notation[Self, PType], NewHelperMixin
):
    __slots__ = ()

    @classmethod
    def _newImpl(self, step: int, acci: Real) -> Self:
        self._step = step
        self._acci = acci
        return self

    @classmethod
    def _fromStepAndAcci(self, step: int, acci: Real) -> Self:
        return self._newHelper(step, acci)


class Pitch[OPType: OPitch](_abc_base.Pitch[OPType], Notation[OPType, Self]):
    __slots__ = ()

    @property
    @lru_cache
    def sgn(self) -> Literal[-1, 0, 1]:
        if self.step == 0:
            return (self.acci > 0) - (self.acci < 0)
        return (self.step > 0) - (self.step < 0)

    @property
    @lru_cache
    def step(self) -> int:
        return self.opitch.step + self.n_steps * self.o

    @property
    def acci(self) -> Real:
        return self.opitch.acci
