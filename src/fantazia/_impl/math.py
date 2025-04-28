from typing import Any, Self
from abc import abstractmethod
from numbers import Integral  # , Rational
# from collections.abc import Iterable, Mapping, Iterator, Sequence
# from collections import Counter
# from functools import lru_cache

# import pyrsistent as pyr

from .utils.cls import classProp, ClassPropMeta
# from .utils.number import primes, nthPrime, prod


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


# class Monzo(Rational, Sequence[int]):
#     """Prime factorization of a rational number."""

#     __slots__ = ("_pf", "_len")

#     def __new__(cls, arg1, *args, n=None, d=None): ...

#     @classmethod
#     def _fromSequence(cls, seq: Iterable[int]):
#         cnt = Counter(seq)
#         length = 0
#         for prime, exponent in zip(primes(), seq):
#             cnt[prime] += exponent
#             length += 1
#         cnt = pyr.pmap(cnt)
#         self = cls._newHelper(cnt)
#         self._len = length
#         return self

#     @classmethod
#     @lru_cache
#     def _fromRational(cls, x: Rational) -> Self:
#         from sympy.ntheory import factorrat

#         factors = factorrat(x)

#     @classmethod
#     def _newHelper(cls, primeFactors: Mapping[int, int]):
#         self = super().__new__(cls)
#         self._pf = primeFactors
#         return self

#     def __iter__(self) -> Iterator[int]:
#         return map(lambda p: self._pf.get(p, 0), primes())

#     def __getitem__(self, index: int) -> int:
#         return self._pf[nthPrime(index)]

#     @property
#     def numerator(self) -> int:
#         return prod(p**e for p, e in self._pf.items() if e > 0)

#     @property
#     def denominator(self) -> int:
#         return prod(p**e for p, e in self._pf.items() if e < 0)

#     def __str__(self) -> str:
#         return f"[{' '.join(map(str, self))}⟩"
