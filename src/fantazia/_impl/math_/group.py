from __future__ import annotations

from numbers import Complex, Integral
from abc import abstractmethod
from functools import lru_cache
from typing import Self, Any

from ..utils.cls import ClassPropMeta, classProp

__all__ = ["AbelianElement", "MulAbelianElement"]

_zero_registry = {}
_one_registry = {}


class AbelianElement(metaclass=ClassPropMeta):
    """
    Represents an element of an **abelian group**, aka. commutative group in terms of the "add"
    operator.
    """

    __slots__ = ()

    @classProp
    @abstractmethod
    def ZERO(cls) -> Self:
        """Identity element of the group."""
        raise NotImplementedError

    @staticmethod
    @lru_cache
    def zero[T: "AbelianElement"](cls: T | type[T]) -> T:
        """
        Returns the zero / identity element of the addition abelian group. This works even
        if `cls` is not an explicit subclass of `AbelianElement`.
        """
        if not isinstance(cls, type):
            cls = cls.__class__
        if cls in _zero_registry:
            return _zero_registry[cls]
        elif hasattr(cls, "ZERO"):
            return cls.ZERO
        return cls(0)

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        """
        `self + other`, which is the fundamental operation of an abelian group.
        """
        raise NotImplementedError

    def __pos__(self) -> Self:
        return self

    @abstractmethod
    def __neg__(self) -> Self:
        """
        `-self`, which refers to the "inverse" of the current element in group theory.
        """
        raise NotImplementedError

    def __sub__(self, other: Any) -> Self:
        """
        `self - other`, equivalent to `self + (-other)`.
        """
        return self + (-other)

    def __mul__(self, other: Any) -> Self:
        """
        `self * other`, only defined when `other` is an integer.

        * If `other` is positive, the result equals to the sum of `other` `self`s
        * If `other` is zero, the result equals to the zero element of the group
        * If `other` is negative, the result equals to the inverse of the sum of `abs(other)` `self`s
        """
        if not isinstance(other, Integral):
            return NotImplemented
        if other == 1:
            return self
        if other == 0:
            if (res := self.ZERO) is NotImplemented:
                res = self - self
            return res
        if other < 0:
            return -((-self) * (-other))
        res = self
        for _ in range(other - 1):
            res += self
        return res

    def __radd__(self, other: Any) -> Self:
        return self.__add__(other)

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)


class MulAbelianElement(metaclass=ClassPropMeta):
    """
    Represents an element of an **abelian group**, aka. commutative group in terms of the
    "multiply" operator.
    """

    __slots__ = ()

    @classProp
    @abstractmethod
    def ONE(cls) -> Self:
        """Identity element of the group."""
        raise NotImplementedError

    @staticmethod
    def one[T: "MulAbelianElement"](cls: T | type[T]) -> T:
        """
        Returns the identity element of a multiplication abelian group type. This works even
        if `cls` is not an explicit subclass of `MulAbelianElement`.
        """
        if not isinstance(cls, type):
            cls = cls.__class__
        if cls in _one_registry:
            return _one_registry[cls]
        elif hasattr(cls, "ONE"):
            return cls.ONE
        return cls(1)

    def __pos__(self) -> Self:
        return self

    @abstractmethod
    def __mul__(self, other: Self) -> Self:
        raise NotImplementedError

    @property
    @abstractmethod
    def inv(self) -> Self:
        """Inversion / reciprocal of the current element."""
        raise NotImplementedError

    def __truediv__(self, other: Any) -> Self:
        if isinstance(other, self.__class__):
            return self * other.inv
        return NotImplemented

    def __pow__(self, other: Any) -> Self:
        if not isinstance(other, Integral):
            return NotImplemented
        if other == 1:
            return self
        if other == 0:
            return self.ONE
        if other < 0:
            return self.inv ** (-other)
        res = self
        for _ in range(other - 1):
            res *= self
        return res

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)

    def __rtruediv__(self, other: Any) -> Self:
        if isinstance(other, MulAbelianElement):
            return other * self.inv
        return NotImplemented


AbelianElement.register(Complex)
MulAbelianElement.register(Complex)
