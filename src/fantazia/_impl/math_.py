from __future__ import annotations

from typing import Any, Self
import typing as t
from abc import abstractmethod
from numbers import Integral, Rational, Real
from collections.abc import Mapping, MutableMapping, Iterator, Callable
from collections import Counter
from functools import lru_cache
from fractions import _hash_algorithm, Fraction as Q
import math
import ast
import operator as op

import pyrsistent as pyr

from .utils.cls import NewHelperMixin
from .utils.cls import (
    cachedProp,
    classProp,
    ClassPropMeta,
    cachedGetter,
    cachedClassProp,
)
from .utils.number import primeFactors, primepi, primes, prod, resolveInt

_hash_algorithm: Callable[[int, int], int]

if t.TYPE_CHECKING:
    import sympy as sp

__all__ = [
    "AbelianElement",
    "MulAbelianElement",
    "Monzo",
]


class AbelianElement(metaclass=ClassPropMeta):
    """
    Represents an element of an **abelian group**, aka commutative group in terms of the "add"
    operator.
    """

    __slots__ = ()

    @classProp
    @abstractmethod
    def ZERO(cls) -> Self:
        """Identity element of the group."""
        raise NotImplementedError

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

    def __radd__(self, other: Any) -> Self:
        return self.__add__(other)

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)


class MulAbelianElement(metaclass=ClassPropMeta):
    __slots__ = ()

    @classProp
    @abstractmethod
    def ONE(cls) -> Self:
        """Identity element of the group."""
        raise NotImplementedError

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


# class Q(Fraction, AbelianElement, MulAbelianElement, metaclass=ClassPropMeta):
#     @cachedClassProp(key="_inf")
#     def INF(cls) -> Self:
#         return cls._from_coprime_ints(1, 0)

#     @cachedClassProp(key="_nan")
#     def NAN(cls) -> Self:
#         return cls._from_coprime_ints(0, 0)

#     @cachedClassProp(key="_zero")
#     def ZERO(cls) -> Self:
#         return cls._from_coprime_ints(0, 1)

#     @cachedClassProp(key="_one")
#     def ONE(cls) -> Self:
#         return cls._from_coprime_ints(1, 1)

#     def __new__(cls, arg0, arg1=None) -> Self:
#         if hasattr(arg0, "__fraction__"):
#             return arg0.__fraction__()
#         return super().__new__(cls, arg0, arg1)

#     @property
#     def inv(self) -> Self:
#         return self._from_coprime_ints(self.denominator, self.numerator)


class Monzo(MulAbelianElement, Real, NewHelperMixin):
    """
    Represents the factorization of a positive simple radical value, in the form of
    $\\prod_{p_i \\text{prime}} p_i^{\\alpha_i}$, where $\\alpha_i$ are rational numbers.
    """

    __slots__ = ("_pf", "_sgn", "_len", "_numerator", "_denominator", "_inv")
    _pf: Mapping[int, Rational]

    @cachedClassProp(key="_one")
    def ONE(cls) -> Self:
        return cls._newHelper(pyr.m())

    def __new__(cls, arg) -> Self:
        if isinstance(arg, cls):
            return arg
        if isinstance(arg, str):
            return cls._parse(arg)
        if isinstance(arg, Rational):
            return cls._fromRational(arg)
        if isinstance(arg, Mapping):
            return cls._newHelper(
                pyr.pmap(primeFactors({k: resolveInt(Q(v)) for k, v in arg.items()}))
            )
        raise ValueError(f"Invalid argument: {arg}.")

    @classmethod
    def _newImpl(cls, pf: Mapping[int, Rational]):
        self = super().__new__(cls)
        self._pf = pf
        return self

    @classmethod
    def _fromMapping(cls, pf: MutableMapping[int, Rational]) -> Self:
        for k in frozenset(k for k, v in pf.items() if v == 0):
            del pf[k]
        if len(pf) == 0:
            return cls.ONE
        return cls._newHelper(pyr.pmap(pf))

    @classmethod
    @lru_cache
    def _fromRational(cls, num: Rational) -> Self:
        return cls._newHelper(pyr.pmap(primeFactors(num)))

    @classmethod
    @lru_cache
    def _parse(self, src: str) -> Self:
        if len(src) == 0:
            return self.ONE
        match ast.parse(src, mode="eval"):
            case ast.Expression(body=body):
                return self._parseAst(body)
            case _:
                raise ValueError(f"Invalid expression: {src}.")

    @classmethod
    @lru_cache
    def _parseAst(cls, src: ast.stmt) -> Self:
        match src:
            case ast.Constant(value=num):
                if not isinstance(num, Integral) or num <= 0:
                    raise ValueError(f"Invalid number: {num}.")
                return cls._fromRational(num)
            case ast.BinOp(left=left, op=ast.Pow(), right=right):
                base = cls._parseAst(left)
                neg = False
                match right:
                    case ast.UnaryOp(op=ast.USub(), operand=right):
                        neg = True
                match right:
                    case ast.Constant(value=exp):
                        if not isinstance(num, Integral):
                            raise ValueError(f"Invalid exponent: {exp}.")
                    case ast.BinOp(
                        left=numer,
                        op=ast.Div(),
                        right=denomin,
                    ):
                        match numer:
                            case ast.Constant(value=p):
                                ...
                            case ast.UnaryOp(
                                op=ast.USub(), operand=ast.Constant(value=p)
                            ):
                                p = -p
                        match denomin:
                            case ast.Constant(value=q):
                                ...
                            case ast.UnaryOp(
                                op=ast.USub(), operand=ast.Constant(value=q)
                            ):
                                q = -q
                        if not isinstance(p, Integral) or not isinstance(q, Integral):
                            raise ValueError(f"Invalid exponent: {ast.unparse(right)}.")
                        exp = resolveInt(Q(p, q))
                    case _:
                        raise ValueError(f"Invalid exponent: {ast.unparse(right)}.")
                if neg:
                    exp = -exp
                return base**exp
            case ast.BinOp(left=left, op=ast.Mult(), right=right):
                num1 = cls._parseAst(left)
                num2 = cls._parseAst(right)
                return num1 * num2
            case ast.BinOp(left=left, op=ast.Div(), right=right):
                num1 = cls._parseAst(left)
                num2 = cls._parseAst(right)
                return num1 / num2
            case _:
                raise ValueError(f"Invalid expression: {ast.unparse(src)}.")

    @property
    def pf(self) -> Mapping[int, Rational]:
        return self._pf

    @cachedProp
    def numerator(self) -> Self:
        res = self._newHelper(pyr.pmap({k: v for k, v in self._pf.items() if v > 0}))
        res._numerator = res
        res._denominator = self.ONE
        return res

    @cachedProp
    def denominator(self) -> Self:
        res = self._newHelper(pyr.pmap({k: -v for k, v in self._pf.items() if v < 0}))
        res._numerator = res
        res._denominator = self.ONE
        return res

    @property
    def rational(self) -> Q:
        """
        Returns a rational number representation of the monzo. If all powers of primes are
        integers, the result is exact. Otherwise, the result equals to the float approximation
        converted to `Fraction`.
        """
        return Q(*self.as_integer_ratio())

    @property
    def value(self) -> int | Q | float:
        if self.is_rational():
            return resolveInt(self.rational)
        else:
            return self._value()

    @cachedGetter
    def __len__(self) -> int:
        return primepi(max(self._pf.keys()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self!s})"

    def __str__(self) -> str:
        return " * ".join(
            str(k) if v == 1 else f"{k}**{v}" if v.is_integer() else f"{k}**({v})"
            for k, v in sorted(self._pf.items())
        )

    def is_rational(self) -> bool:
        return all(v.is_integer() for v in self._pf.values())

    def is_integer(self) -> bool:
        return all(v >= 0 and v.is_integer() for v in self._pf.values())

    def __abs__(self) -> Self:
        return self

    def __neg__(self) -> Q | float:
        if self.is_rational():
            return -self.rational
        else:
            return -float(self)

    def __mul__(self, other: Any) -> Self:
        if isinstance(other, Rational):
            if other > 0:
                other = self._fromRational(other)
            else:
                if self.is_rational():
                    return self.rational * other
                else:
                    return float(self) * other
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._merge(other, op.add)

    def __truediv__(self, other: Any):
        if isinstance(other, Rational):
            if other > 0:
                other = self._fromRational(other)
            else:
                if self.is_rational():
                    return self.rational / other
                else:
                    return float(self) / other
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._merge(other, op.sub)

    @cachedProp
    def inv(self) -> Self:
        res = self._newHelper(pyr.pmap({k: -v for k, v in self._pf.items()}))
        res._inv = self
        if hasattr(self, "_numerator"):
            res._denominator = self._numerator
        if hasattr(self, "_denominator"):
            res._numerator = self._denominator
        return res

    def __pow__(self, other: Any) -> Self:
        if other == 1:
            return self
        if other == 0:
            return self.ONE
        if isinstance(other, self.__class__) and other.is_rational():
            other = other.rational
        if not isinstance(other, Rational):
            return NotImplemented
        other = resolveInt(Q(other))
        pf = pyr.pmap({k: v * other for k, v in self._pf.items()})
        return self._newHelper(pf)

    def __iter__(self) -> Iterator[int]:
        for p in primes(len(self)):
            return self._pf[p]

    def _value(self) -> int | float:
        return prod(k**v for k, v in self._pf.items())

    def __int__(self) -> int:
        return int(self._value())

    def __float__(self) -> float:
        return float(self._value())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self._pf == other._pf
        elif isinstance(other, Rational):
            if other <= 0:
                return False
            other = self._fromRational(other)
            return self._pf == other._pf
        elif isinstance(other, Real):
            return float(self) == float(other)
        return False

    def __hash__(self) -> int:
        # hash must be consistent with builtin number types
        if self.is_integer():
            return hash(int(self))
        elif self.is_rational():
            return _hash_algorithm(int(self.numerator), int(self.denominator))
        else:
            return hash(self._pf)

    def as_integer_ratio(self) -> tuple[int, int]:
        if self.is_integer():
            return int(self), 1
        elif self.is_rational():
            return int(self.numerator), int(self.denominator)
        else:
            return float(self).as_integer_ratio()

    # TODO)) implement abstract methods of `Real`

    def _add_sub(
        self, other: Any, op: Callable[[Rational, Rational], Rational]
    ) -> Self | Q | float:
        if other == 0:
            return self
        if isinstance(other, Rational):
            if self.is_rational():
                result = Q(op(self.rational, other))
                if result <= 0:
                    return result
                else:
                    return self._fromRational(result)
            other = self._fromRational(other)
        if isinstance(other, self.__class__):
            gcd = self.gcd(other)
            k1 = self / gcd
            k2 = other / gcd
            if k1.is_rational() and k2.is_rational():
                k = Q(op(k1.rational, k2.rational))
                if k <= 0:
                    return k * float(self)
                else:
                    k = self._fromRational(k)
                    return k * gcd
        return op(float(self), float(other))

    def __add__(self, other: Any) -> Self | Q | float:
        return self._add_sub(other, op.add)

    def __radd__(self, other) -> Self | Q | float:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Self:
        return self._add_sub(other, op.sub)

    def __ceil__(self) -> int:
        return math.ceil(self._value())

    def __floor__(self) -> int:
        return math.floor(self._value())

    def __floordiv__(self, other: Any) -> int:
        if isinstance(other, (self.__class__, Rational)):
            return math.floor(self / other)
        return int(float(self) // other)

    def __le__(self, other) -> bool:
        return self._value() <= other

    def __lt__(self, other) -> bool:
        return self._value() < other

    def __mod__(self, other):
        raise NotImplementedError

    def __rfloordiv__(self, other):
        raise NotImplementedError

    def __rmod__(self, other):
        raise NotImplementedError

    def __round__(self, ndigits: Integral | None = None) -> int | float:
        if ndigits is None:
            return round(self._value())
        else:
            if self.is_rational():
                return self._fromRational(round(self.rational, ndigits))
            else:
                ...  # TODO))

    def __rpow__(self, base: Any):
        if self.is_rational():
            return base**self.rational
        else:
            return base ** float(self)

    def __trunc__(self) -> int:
        return math.floor(self)

    def _merge(self, other: Self, fn: Callable[[Rational, Rational], Rational]) -> Self:
        pf = Counter()
        for k in self._pf.keys() | other._pf.keys():
            newExp = fn(self._pf.get(k, 0), other._pf.get(k, 0))
            if newExp != 0:
                pf[k] = newExp
        pf = pyr.pmap(pf)
        return self._newHelper(pf)

    def gcd(self, other: Self | Rational) -> Self:
        if not isinstance(other, self.__class__):
            other = self._fromRational(other)
        return self._merge(other, min)

    def lcm(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            other = self._fromRational(other)
        return self._merge(other, max)

    def sympy(self) -> sp.Expr:
        """
        Convert the monzo to a `sympy` expression.
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError(
                "Conversion to `sympy` expression requires `sympy` to be installed."
            )

        return prod(sp.Integer(k) ** v for k, v in self._pf.items())

    # def _format_gen(
    #     self,
    #     *,
    #     space: bool = True,
    #     div: bool = False,
    #     exp1: bool = False,
    # ) -> Iterator[str]:
    #     for prime, exp in sorted(self._pf.items()):
    #         if div and exp < 0:
    #             ...
