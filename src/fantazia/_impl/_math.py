from typing import Any, Self
from abc import abstractmethod
from numbers import Integral

from .utils.cls import classProp, ClassPropMeta


class AbelianElement(metaclass=ClassPropMeta):
    """
    Represents an element of an **abelian group**, aka commutative group.
    """

    __slots__ = ()

    @classProp
    @abstractmethod
    def ZERO(cls) -> Self:
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        """
        `self + other`, which is the fundamental operation of an abelian group.
        """
        raise NotImplementedError

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
        `self * other`, olny defined when `other` is an integer.

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

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)
