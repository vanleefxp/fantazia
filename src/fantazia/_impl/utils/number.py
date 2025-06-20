from __future__ import annotations

from collections.abc import Sequence, Callable, Iterable, Iterator, Mapping
from collections import Counter
import typing as t
from typing import overload, Literal, Self
from enum import StrEnum
from bisect import bisect_left
from numbers import Number, Real, Rational, Integral
from fractions import Fraction as Q
from math import inf, nan
import itertools as it
import sys
from .cls import singleton
from .._snippet.sympy_primepi import primepi

if t.TYPE_CHECKING:
    import numpy as np
    from _typeshed import SupportsRichComparison

__all__ = [
    "nextPow2",
    "prevPow2",
    "RMode",
    "rdivmod",
    "rdiv",
    "rmod",
    "rbisect",
    "gprev",
    "gnext",
    "fdiv",
    "cdiv",
    "grange",
    "smod",
    "resolveInt",
    "clamp",
    "interpExp",
]

if t.TYPE_CHECKING:

    @overload
    def nextPow2(n: int, strict: bool, withPow: Literal[True]) -> tuple[int, int]: ...

    @overload
    def nextPow2(n: int, strict: bool, withPow: Literal[False]) -> int: ...

    @overload
    def prevPow2(n: int, strict: bool, withPow: Literal[True]) -> tuple[int, int]: ...

    @overload
    def prevPow2(n: int, strict: bool, withPow: Literal[False]) -> int: ...

    @overload
    def resolveInt[N: Number](arg: N) -> int | N: ...

    @overload
    def resolveInt[N: Number](arg: Callable[..., N]) -> Callable[..., int | N]: ...

    @overload
    def qdiv(a: Rational, b: Rational) -> Rational: ...

    @overload
    def qdiv(a: Real, b: Real) -> Real: ...

    @overload
    def pf2Rational(*pows: int) -> Q: ...

    @overload
    def pf2Rational(pows: Iterable[int] | Mapping[int, int]) -> Q: ...

    @overload
    def minmax[T: SupportsRichComparison](*items: T) -> tuple[T, T]: ...

    @overload
    def minmax[T: SupportsRichComparison](items: Iterable[T]) -> tuple[T, T]: ...

    @overload
    def primes(stop: Integral) -> Iterator[int]: ...

    @overload
    def primes(start: Integral, stop: Integral) -> Iterator[int]: ...


def nextPow2(
    n: int, strict: bool = True, withPow: bool = False
) -> tuple[int, int] | int:
    if n <= 0:
        return 1
    result = 1
    power = 0
    if strict:
        while result <= n:
            result <<= 1
            power += 1
    else:
        while result < n:
            result <<= 1
            power += 1
    if withPow:
        return result, power
    else:
        return result


def prevPow2(
    n: int, strict: bool = True, withPow: bool = False
) -> int | tuple[int, int]:
    result, power = nextPow2(n, False)
    if strict:
        while result >= n:
            result >>= 1
            power -= 1
    else:
        while result > n:
            result >>= 1
            power -= 1
    if withPow:
        return result, power
    else:
        return result


def gnext(n: float, k: float, rem: float = 0, strict: bool = True) -> float:
    """
    Returns the smallest number in the form of `k * t + r` greater than `n` where `t` is an
    integer.
    """
    if n % k == rem:
        return n + k if strict else n
    return ((n - rem) // k + 1) * k + rem


def gprev(n: float, k: float, rem: float = 0, strict: bool = True) -> float:
    """
    Returns the largest number in the form of `k * t + r` less than `n` where `t` is an integer.
    """
    if n % k == rem:
        return n - k if strict else n
    return (n - rem) // k * k + rem


def fdiv(n: float, k: float, strict: bool = False) -> int:
    if strict and n % k == 0:
        return int(n // k) - 1
    return int(n // k)


def cdiv(n: float, k: float, strict: bool = False) -> int:
    if strict and n % k == 0:
        return int(n // k) + 1
    return int(-(-n // k))


def grange(
    start: float,
    stop: float,
    k: float,
    rem: float = 0,
    includeStart: bool = True,
    includeEnd: bool = True,
) -> Iterable[float]:
    """
    Yields all integer multiples of `k` with remainder `rem` in the range `[start, stop]`.
    """
    return map(
        lambda t: t * k + rem,
        range(
            cdiv(start - rem, k, not includeStart),
            fdiv(stop - rem, k, not includeEnd) + 1,
        ),
    )


def smod(n: float, k: float, disp: float, includeRight: bool = True) -> float:
    result = n % k
    if result > disp or (not includeRight and result == disp):
        result -= k
    return result


class RMode(StrEnum):
    """
    Rounding mode identifiers. The values are inspired by the Python standard library
    `decimal`.
    """

    FLOOR = F = "f"
    """Round towards negative infinity."""

    CEIL = C = "c"
    """Round towards positive infinity."""

    UP = U = "u"
    """Round away from zero."""

    DOWN = D = "d"
    """Round towards zero."""

    EVEN = E = "e"
    """Round to the nearest even integer."""

    ODD = O = "o"  # noqa: E741
    """Round to the nearest odd integer."""

    def __neg__(self) -> Self:
        if self is RMode.FLOOR:
            return RMode.CEIL
        elif self is RMode.CEIL:
            return RMode.FLOOR
        else:
            return self


def rdivmod[N: Real](
    n: N, d: N = 1, /, round: bool = True, rmode: RMode | str = RMode.E, thresh: N = 0
) -> tuple[int, N]:
    """
    `divmod()` with rounding support.

    The return result is a tuple `(q, r)` satisfying the invariant `n == q * d + r`. The
    inequality `abs(r) < abs(d)` always holds.
    """
    rmode = RMode(rmode)
    q, r = divmod(n, d)
    q = int(q)
    if r == 0:
        return (q, 0)
    if not round or r * 2 == d:
        match rmode:
            case RMode.FLOOR:
                pass
            case RMode.CEIL:
                q += 1
                r -= d
            case RMode.UP:
                if q >= thresh:
                    q += 1
                    r -= d
            case RMode.DOWN:
                if q < thresh:
                    q += 1
                    r -= d
            case RMode.EVEN:
                if q % 2 == 1:
                    q += 1
                    r -= d
            case RMode.ODD:
                if q % 2 == 0:
                    q += 1
                    r -= d
            case _:
                raise ValueError(f"Invalid round mode: {rmode}")
    elif r * 2 > d:
        q += 1
        r -= d
    return (q, r)


def rdiv[N: Real](n: N, d: N = 1, **kwargs) -> int:
    return rdivmod(n, d, **kwargs)[0]


def rmod[N: Real](n: N, d: N = 1, **kwargs) -> N:
    return rdivmod(n, d, **kwargs)[1]


_constFunc = lambda x: x  # noqa: E731


def cmp[T](a: T, b: T, /, key: Callable[[T], int] | None = None):
    if key is None:
        return (a > b) - (a < b)
    else:
        return (key(a) > key(b)) - (key(a) < key(b))


def rbisect[T](
    a: Sequence[T],
    x: T,
    lo: int | None = 0,
    hi: int | None = None,
    /,
    key: Callable[[T], int] | None = None,
    round: bool = True,
    rmode: RMode | str = RMode.E,
) -> int:
    """Binary search with rounding support."""

    rmode = RMode(rmode)
    if rmode is RMode.D:
        rmode = RMode.F
    elif rmode is RMode.U:
        rmode = RMode.C
    idx = bisect_left(a, x, lo, hi, key=key)
    if key is None:
        key = _constFunc

    length = len(a)
    if idx >= length:  # exceeds right boundary
        if round:
            return length - 1
        else:
            match rmode:
                case RMode.F:
                    return length - 1
                case RMode.C:
                    return length
                case RMode.E:
                    return length - 1 if length % 2 == 1 else length
                case RMode.O:
                    return length if length % 2 == 1 else length - 1
                case _:
                    raise ValueError(f"Invalid round mode: {rmode}")
    if key(a[idx]) == key(x):  # element contained in the list
        return idx
    if idx == 0:  # element smaller than the left boundary
        if round:
            return 0
        else:
            match rmode:
                case RMode.F:
                    return -1
                case RMode.C:
                    return 0
                case RMode.E:
                    return 0 if length % 2 == 1 else -1
                case RMode.O:
                    return -1 if length % 2 == 1 else 0
                case _:
                    raise ValueError(f"Invalid round mode: {rmode}")
    if round:
        dl = key(x) - key(a[idx - 1])
        dr = key(a[idx]) - key(x)
        if dl < dr:
            return idx - 1
        elif dl > dr:
            return idx
    match rmode:
        case RMode.F:
            return idx - 1
        case RMode.C:
            return idx
        case RMode.E:
            return idx - 1 if idx % 2 == 1 else idx
        case RMode.O:
            return idx if idx % 2 == 1 else idx - 1
        case _:
            raise ValueError(f"Invalid round mode: {rmode}")


def bsearch[T](
    a: Sequence[T],
    x: T,
    lo: int | None = 0,
    hi: int | None = None,
    key: Callable[[T], int] | None = None,
) -> int:
    """
    Perform binary search on a sorted sequence `a` to find the index of item `x`.
    If `x` exists in `a`, returns its index. Otherwise, returns -(insertion point) - 1

    **Note**: This method works like Java's `Arrays.binarySearch()` method.
    """
    idx = bisect_left(a, x, lo, hi, key)
    item = a[idx]
    if item == x:
        return idx
    else:
        return -idx - 1


def resolveInt(arg: Number | Callable[..., Number]) -> Number | Callable[..., Number]:
    """
    Turns a numeric value into an integer if it is an integer, otherwise returns the value as
    is. This function can also be used as a decorator.
    """

    if callable(arg):

        def wrapper(*args, **kwargs) -> int | float:
            result = arg(*args, **kwargs)
            if result.is_integer():
                return int(result)
            else:
                return result

        return wrapper
    else:
        if arg.is_integer():
            return int(arg)
        else:
            return arg


def qdiv(a: Real, b: Real) -> Real:
    """
    Division that returns an exact value when both dividend and divisor are rational numbers.
    """
    if b == 1:
        return a
    if isinstance(a, Rational) and isinstance(b, Rational):
        return resolveInt(Q(a, b))
    return a / b


def safeDiv(a: float, b: float) -> float:
    if b == 0:
        if a > 0:
            return inf
        elif a < 0:
            return -inf
        else:
            return nan
    else:
        return a / b


def clamp[T: SupportsRichComparison](a: T, c: T, b: T) -> T:
    if c < a:
        return a
    elif b < c:
        return b
    else:
        return c


def interpExp(a: float, b: float, t: float) -> float:
    la = np.log(a)
    lb = np.log(b)
    return np.exp(la * (1 - t) + lb * t)


def _clusterCenters(data: "np.ndarray", labels) -> "np.ndarray":
    import numpy as np

    nClusters = np.max(labels) + 1
    centers = np.empty(nClusters, dtype=float)
    for i in range(nClusters):
        cluster = data[labels == i]
        centers[i] = np.mean(cluster)
    return centers


def approxGCD(data: Iterable[float], tolerance: float) -> float:
    """
    find a value `k` such that all data elements are approximately multiples of `k`
    """
    from sklearn.cluster import DBSCAN
    import numpy as np

    dbscan = DBSCAN(eps=tolerance, min_samples=1)
    data = np.array(data, dtype=float)
    while len(data) > 1:
        # remove elements close to zero
        # otherwise the algorithm might never terminate
        data = data[abs(data) > tolerance]
        data.sort()
        data = data - np.insert(data, 0, 0)[:-1]
        clusterResult = dbscan.fit(data.reshape(-1, 1))
        labels = clusterResult.labels_
        data = _clusterCenters(data, labels)
    return data[0]


_PRIME_TABLE_MAX_SIZE = 10000


@singleton
class _PrimeTable:
    def __new__(cls):
        self = super().__new__(cls)
        self._data = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        return self

    def __getitem__(self, n: int) -> int:
        from gmpy2 import next_prime

        large_n = n > _PRIME_TABLE_MAX_SIZE
        bound = min(n, _PRIME_TABLE_MAX_SIZE)
        if len(self._data) <= bound:
            lastPrime = self._data[-1]
            while len(self._data) <= bound:
                lastPrime = next_prime(lastPrime)
                self._data.append(lastPrime)
        if large_n:
            remaining = n - _PRIME_TABLE_MAX_SIZE
            prime = self._data[-1]
            for _ in range(remaining):
                prime = next_prime(prime)
            return prime
        return self._data[n]


def primes(a=None, b=None, /, *, start=None, stop=None, n=None) -> Iterable[int]:
    """A generator that yields prime numbers."""
    from gmpy2 import next_prime

    if b is None:
        if a is None:
            if start is None:
                start = 2
        elif n is not None and start is None:
            start = a
        elif stop is None:
            stop = a
    else:
        if start is None:
            start = a
        if stop is None:
            stop = b

    num = int(next_prime(start - 1))
    if n is not None:
        for _ in range(n):
            yield num
            num = int(next_prime(num))
    elif stop is None:
        while True:
            yield num
            num = int(next_prime(num))
    else:
        while num < stop:
            yield num
            num = int(next_prime(num))


def prime(n: int) -> int:
    """
    Returns the `n`-th prime number.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return _PrimeTable()[n]


def alternateSignInts() -> Iterator[int]:
    """
    Generates the sequence 0, 1, -1, 2, -2, ... This demonstrates how the set of integers
    has the same cardinality as the set of natural numbers.
    """
    yield 0
    for i in it.count(1):
        yield i
        yield -i


def pf2Rational(arg0, *args):
    """
    Convert the prime factorization of a rational number to its fraction representation.
    """
    if isinstance(arg0, Mapping):
        n = d = 1
        for prime, pow in arg0.items():
            if pow > 0:
                n *= prime**pow
            elif pow < 0:
                d *= prime**-pow
        return Q(n, d)

    if isinstance(arg0, Iterable):
        pows = arg0
    else:
        pows = it.chain((arg0,), args)
    n = d = 1
    for pow, prime in zip(pows, primes()):
        if pow > 0:
            n *= prime**pow
        elif pow < 0:
            d *= prime**-pow
    return Q(n, d)


def _primeFactors(num: Rational) -> Counter[int]:
    from primefac import primefac

    pf = Counter(primefac(num.numerator))
    pf.subtract(primefac(num.denominator))
    return pf


def _cleanZeroes(pf: Counter[int]) -> Counter[int]:
    """
    Remove all zero powers from the prime factorization.
    """
    for k, v in tuple(pf.items()):
        if v == 0:
            del pf[k]
    return pf


def primeFactors(*args) -> Counter[int]:
    """
    Factorize a rational number or simple radical into powers of prime.
    """
    if len(args) == 1 and isinstance(args[0], Real):
        pf = _primeFactors(args[0])
        return pf
    else:
        pf = Counter()
        for arg in args:
            if isinstance(arg, Mapping):
                for k, v in arg.items():
                    if v != 0:
                        pf_k = _primeFactors(k)
                        for k1, v1 in pf_k.items():
                            pf[k1] += v * v1
            elif isinstance(arg, Iterable):
                for num in arg:
                    pf.update(_primeFactors(num))
            else:
                if pf is None:
                    pf = _primeFactors(arg)
                else:
                    pf.update(_primeFactors(arg))
    return _cleanZeroes(pf)


def powerOf(num: Rational, base: Rational) -> int:
    p = 0
    while num % base == 0:
        num //= base
        p += 1
    return p


def minmax(arg0, *args):
    if len(args) == 0 and isinstance(arg0, Iterable):
        itr = iter(arg0)
        min_item = max_item = next(itr)
        for item in itr:
            if item < min_item:
                min_item = item
            elif item > max_item:
                max_item = item
        return min_item, max_item

    min_item = max_item = arg0
    for item in args:
        if item < min_item:
            min_item = item
        elif item > max_item:
            max_item = item
    return min_item, max_item


def resolveNpType(num):
    import numpy as np

    if num.is_integer():
        return int(num)
    elif isinstance(num, np.floating):
        return float(num)
    else:
        return num


if "sympy" in sys.modules:
    # use `sympy`'s factorization function if available
    # sympy is already imported, so importing again will not cost much time
    import sympy as sp
    from sympy.ntheory.generate import prime, _primepi as primepi  # noqa: F401

    primepi: Callable[[Integral], int]

    def _primeFactors(num: Rational) -> Counter[int]:
        return Counter(sp.factorrat(sp.Rational(num)))
