from __future__ import annotations

from typing import Any, Self
from types import MappingProxyType as mappingproxy
import typing as t
from abc import abstractmethod, ABCMeta
from numbers import Integral, Rational, Real, Complex
from collections.abc import Mapping, MutableMapping, Iterable, Iterator, Callable
from collections import Counter
from functools import lru_cache
from fractions import _hash_algorithm, Fraction as Q
import math
import ast
import operator as op
import warnings
import itertools as it

import pyrsistent as pyr

from .utils.cls import NewHelperMixin
from .utils.cls import (
    cachedProp,
    classProp,
    ClassPropMeta,
    cachedGetter,
    cachedClassProp,
)
from .utils.number import primeFactors, primepi, primes, prime, resolveInt

_hash_algorithm: Callable[[int, int], int]

if t.TYPE_CHECKING:
    import sympy as sp

__all__ = [
    "AbelianElement",
    "MulAbelianElement",
    "Monzo",
]

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


class CounterBase[T, N: Real](Mapping[T, N], metaclass=ABCMeta):
    """
    Abstract base class for counter-like mappings. Provides an abstraction over the
    `Counter` class from the `collections` module, which is a subclass of `dict`.
    """

    # code partly copied from `collections.Counter`

    def total(self) -> N:
        "Sum of the counts"
        return sum(self.values())

    def most_common(self, n: Integral | None = None):
        """
        List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.
        """
        # Emulate Bag.sortedByCount from Smalltalk
        if n is None:
            return sorted(self.items(), key=op.itemgetter(1), reverse=True)

        # Lazy import to speedup Python startup time
        import heapq

        return heapq.nlargest(n, self.items(), key=op.itemgetter(1))

    def elements(self) -> Iterator[T]:
        """
        Iterator over elements repeating each as many times as its count.

        **Note**: This method only works when all counts are integers. If an element's count
        has been set to zero or is a negative number, `elements()` will ignore it.
        """
        # Emulate Bag.do from Smalltalk and Multiset.begin from C++.
        return it.chain.from_iterable(it.starmap(it.repeat, self.items()))

    def __eq__(self, other: Any) -> bool:
        "True if all counts agree. Missing counts are treated as zero."
        if not isinstance(other, CounterBase):
            return NotImplemented
        return all(self[e] == other[e] for c in (self, other) for e in c)

    def __ne__(self, other: Any) -> bool:
        "True if any counts disagree. Missing counts are treated as zero."
        if not isinstance(other, CounterBase):
            return NotImplemented
        return not self == other

    def __le__(self, other: Any) -> bool:
        "True if all counts in self are a subset of those in other."
        if not isinstance(other, CounterBase):
            return NotImplemented
        return all(self[e] <= other[e] for c in (self, other) for e in c)

    def __lt__(self, other: Any) -> bool:
        "True if all counts in self are a proper subset of those in other."
        if not isinstance(other, CounterBase):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other: Any) -> bool:
        "True if all counts in self are a superset of those in other."
        if not isinstance(other, CounterBase):
            return NotImplemented
        return all(self[e] >= other[e] for c in (self, other) for e in c)

    def __gt__(self, other: Any) -> bool:
        "True if all counts in self are a proper superset of those in other."
        if not isinstance(other, CounterBase):
            return NotImplemented
        return self >= other and self != other


class MutableCounterBase[T, N: Real](
    CounterBase[T, N], MutableMapping[T, N], metaclass=ABCMeta
):
    def update(self, iterable: Mapping[T, N] | Iterable[T] | None = None, /, **kwds):
        """Like dict.update() but add counts instead of replacing them.

        Source can be an iterable, a dictionary, or another Counter instance.

        >>> c = Counter('which')
        >>> c.update('witch')           # add elements from another iterable
        >>> d = Counter('watch')
        >>> c.update(d)                 # add elements from another counter
        >>> c['h']                      # four 'h' in which, witch, and watch
        4

        """
        # The regular dict.update() operation makes no sense here because the
        # replace behavior results in some of the original untouched counts
        # being mixed-in with all of the other counts for a mismash that
        # doesn't have a straight-forward interpretation in most counting
        # contexts.  Instead, we implement straight-addition.  Both the inputs
        # and outputs are allowed to contain zero and negative counts.

        if iterable is not None:
            if isinstance(iterable, Mapping):
                if self:
                    self_get = self.get
                    for elem, count in iterable.items():
                        self[elem] = count + self_get(elem, 0)
                else:
                    # fast path when counter is empty
                    super().update(iterable)
            else:
                from collections import _count_elements

                _count_elements(self, iterable)
        if kwds:
            self.update(kwds)

    def subtract(self, iterable=None, /, **kwds):
        """Like dict.update() but subtracts counts instead of replacing them.
        Counts can be reduced below zero.  Both the inputs and outputs are
        allowed to contain zero and negative counts.

        Source can be an iterable, a dictionary, or another Counter instance.

        >>> c = Counter('which')
        >>> c.subtract('witch')             # subtract elements from another iterable
        >>> c.subtract(Counter('watch'))    # subtract elements from another counter
        >>> c['h']                          # 2 in which, minus 1 in witch, minus 1 in watch
        0
        >>> c['w']                          # 1 in which, minus 1 in witch, minus 1 in watch
        -1

        """
        if iterable is not None:
            self_get = self.get
            if isinstance(iterable, Mapping):
                for elem, count in iterable.items():
                    self[elem] = self_get(elem, 0) - count
            else:
                for elem in iterable:
                    self[elem] = self_get(elem, 0) - 1
        if kwds:
            self.subtract(kwds)

    def clean_zeroes(self):
        for k in frozenset(k for k, v in self.items() if v == 0):
            del self[k]

    def keep_positive(self):
        """Internal method to strip elements with a negative or zero count"""
        nonpositive = [elem for elem, count in self.items() if not count > 0]
        for elem in nonpositive:
            del self[elem]
        return self

    def __iadd__(self, other):
        """Inplace add from another counter, keeping only positive counts.

        >>> c = Counter('abbb')
        >>> c += Counter('bcc')
        >>> c
        Counter({'b': 4, 'c': 2, 'a': 1})

        """
        for elem, count in other.items():
            self[elem] += count
        return self.keep_positive()

    def __isub__(self, other):
        """Inplace subtract counter, but keep only results with positive counts.

        >>> c = Counter('abbbc')
        >>> c -= Counter('bccd')
        >>> c
        Counter({'b': 2, 'a': 1})

        """
        for elem, count in other.items():
            self[elem] -= count
        return self.keep_positive()

    def __ior__(self, other):
        """Inplace union is the maximum of value from either counter.

        >>> c = Counter('abbb')
        >>> c |= Counter('bcc')
        >>> c
        Counter({'b': 3, 'c': 2, 'a': 1})

        """
        for elem, other_count in other.items():
            count = self[elem]
            if other_count > count:
                self[elem] = other_count
        return self.keep_positive()

    def __iand__(self, other):
        """Inplace intersection is the minimum of corresponding counts.

        >>> c = Counter('abbb')
        >>> c &= Counter('bcc')
        >>> c
        Counter({'b': 1})

        """
        for elem, count in self.items():
            other_count = other[elem]
            if other_count < count:
                self[elem] = other_count
        return self.keep_positive()


MutableCounterBase.register(Counter)


class CounterView[T, N: Real](CounterBase[T, N]):
    def __init__(self, data: Mapping[T, N]):
        super().__init__()
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key) -> N:
        return super().get(key, 0)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)


class MutableCounterView[T, N: AbelianElement](
    CounterView[T, N], MutableCounterBase[T, N]
): ...


class MonzoBase(metaclass=ABCMeta):
    __slots__ = ()

    @property
    @abstractmethod
    def pf(self) -> Mapping[int, Rational]:
        raise NotImplementedError

    def __len__(self) -> int:
        return primepi(max(self.pf.keys()))

    def __getitem__(self, key: int) -> Rational:
        return self.pf[prime(key)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self!s})"

    def __str__(self) -> str:
        return " * ".join(
            str(k) if v == 1 else f"{k}**{v}" if v.is_integer() else f"{k}**({v})"
            for k, v in sorted(self.pf.items())
        )

    def is_rational(self) -> bool:
        return all(v.is_integer() for v in self.pf.values())

    def is_integer(self) -> bool:
        return all(v >= 0 and v.is_integer() for v in self.pf.values())

    def __iter__(self) -> Iterator[int]:
        for p in primes(len(self)):
            yield self.pf[p]

    def _value(self) -> int | float:
        return math.prod(k**v for k, v in self._pf.items())

    def __int__(self) -> int:
        return int(self._value())

    def __float__(self) -> float:
        return float(self._value())


def _counter_cleanZeroes[T, N: Real](m: MutableMapping[T, N]) -> MutableMapping[T, N]:
    for k in frozenset(k for k, v in m.items() if v == 0):
        del m[k]
    return m


def _counter_update[T, N: Real](
    m: MutableMapping[T, N], other: Mapping[T, N] | Iterable[T]
) -> MutableMapping[T, N]:
    Counter.update(m, other)
    _counter_cleanZeroes(m)
    return m


def _counter_subtract[T, N: Real](
    m: MutableMapping[T, N], other: Mapping[T, N] | Iterable[T]
) -> MutableMapping[T, N]:
    Counter.subtract(m, other)
    _counter_cleanZeroes(m)
    return m


class MutableMonzo(MonzoBase):
    __slots__ = ("_pf", "_pf_proxy")

    def __init__(self):
        super().__init__()
        self._pf = dict()

    @cachedProp(key="_pf_proxy")
    def pf(self) -> Mapping[int, Rational]:
        return mappingproxy(self._pf)

    def __imul__(
        self, other: Mapping[Integral, Rational] | MonzoBase | Rational
    ) -> Self:
        if isinstance(other, MonzoBase):
            _counter_update(self._pf, other.pf)
            return self
        if isinstance(other, Rational):
            if other <= 0:
                raise ValueError("Negative value not allowed.")
            p, q = other.as_integer_ratio()

            from primefac import primefac

            Counter.update(self._pf, primefac(p))
            Counter.subtract(self._pf, primefac(q))
            _counter_cleanZeroes(self._pf)

            return self

        return NotImplemented


# TODO)) implement a mutable monzo type to speed up monzo creation


class Monzo(MonzoBase, MulAbelianElement, Real, NewHelperMixin):
    """
    Represents the factorization of a positive simple radical value, in the form of
    $\\prod_{p_i \\text{prime}} p_i^{\\alpha_i}$, where $\\alpha_i$ are rational numbers.
    """

    __slots__ = ("_pf", "_sgn", "_len", "_numerator", "_denominator", "_inv", "_hash")
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
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=num)):
                if not isinstance(num, Integral):
                    raise ValueError(f"Invalid number: {num}.")
                warnings.warn(
                    "Monzo represents positive values only. Here the absolute value is taken."
                )
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

    @cachedGetter
    def __len__(self) -> int:
        return super().__len__()

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

    @cachedGetter
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

        return math.prod(sp.Integer(k) ** v for k, v in self._pf.items())

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
