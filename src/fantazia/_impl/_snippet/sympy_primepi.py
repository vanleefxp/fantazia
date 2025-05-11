# snippet borrowed from `sympy`

from bisect import bisect, bisect_left
from array import array
import math
import itertools as it

__all__ = ["primepi"]


class Sieve:
    """A list of prime numbers, implemented as a dynamically
    growing sieve of Eratosthenes. When a lookup is requested involving
    an odd number that has not been sieved, the sieve is automatically
    extended up to that number. Implementation details limit the number of
    primes to ``2^32-1``.

    Examples
    ========

    >>> from sympy import sieve
    >>> sieve._reset() # this line for doctest only
    >>> 25 in sieve
    False
    >>> sieve._list
    array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])
    """

    # data shared (and updated) by all Sieve instances
    def __init__(self, sieve_interval=1_000_000):
        """Initial parameters for the Sieve class.

        Parameters
        ==========

        sieve_interval (int): Amount of memory to be used

        Raises
        ======

        ValueError
            If ``sieve_interval`` is not positive.

        """
        self._n = 6
        self._list = array("L", [2, 3, 5, 7, 11, 13])  # primes
        self._tlist = array("L", [0, 1, 1, 2, 2, 4])  # totient
        self._mlist = array("i", [0, 1, -1, -1, 0, -1])  # mobius
        if sieve_interval <= 0:
            raise ValueError("sieve_interval should be a positive integer")
        self.sieve_interval = sieve_interval
        assert all(len(i) == self._n for i in (self._list, self._tlist, self._mlist))

    def __repr__(self):
        return (
            "<%s sieve (%i): %i, %i, %i, ... %i, %i\n"
            "%s sieve (%i): %i, %i, %i, ... %i, %i\n"
            "%s sieve (%i): %i, %i, %i, ... %i, %i>"
        ) % (
            "prime",
            len(self._list),
            self._list[0],
            self._list[1],
            self._list[2],
            self._list[-2],
            self._list[-1],
            "totient",
            len(self._tlist),
            self._tlist[0],
            self._tlist[1],
            self._tlist[2],
            self._tlist[-2],
            self._tlist[-1],
            "mobius",
            len(self._mlist),
            self._mlist[0],
            self._mlist[1],
            self._mlist[2],
            self._mlist[-2],
            self._mlist[-1],
        )

    def _reset(self, prime=None, totient=None, mobius=None):
        """Reset all caches (default). To reset one or more set the
        desired keyword to True."""
        if all(i is None for i in (prime, totient, mobius)):
            prime = totient = mobius = True
        if prime:
            self._list = self._list[: self._n]
        if totient:
            self._tlist = self._tlist[: self._n]
        if mobius:
            self._mlist = self._mlist[: self._n]

    def extend(self, n):
        """Grow the sieve to cover all primes <= n.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend(30)
        >>> sieve[10] == 29
        True
        """
        n = int(n)
        # `num` is even at any point in the function.
        # This satisfies the condition required by `self._primerange`.
        num = self._list[-1] + 1
        if n < num:
            return
        num2 = num**2
        while num2 <= n:
            self._list += array("L", self._primerange(num, num2))
            num, num2 = num2, num2**2
        # Merge the sieves
        self._list += array("L", self._primerange(num, n + 1))

    def _primerange(self, a, b):
        """Generate all prime numbers in the range (a, b).

        Parameters
        ==========

        a, b : positive integers assuming the following conditions
                * a is an even number
                * 2 < self._list[-1] < a < b < nextprime(self._list[-1])**2

        Yields
        ======

        p (int): prime numbers such that ``a < p < b``

        Examples
        ========

        >>> from sympy.ntheory.generate import Sieve
        >>> s = Sieve()
        >>> s._list[-1]
        13
        >>> list(s._primerange(18, 31))
        [19, 23, 29]

        """
        if b % 2:
            b -= 1
        while a < b:
            block_size = min(self.sieve_interval, (b - a) // 2)
            # Create the list such that block[x] iff (a + 2x + 1) is prime.
            # Note that even numbers are not considered here.
            block = [True] * block_size
            for p in self._list[
                1 : bisect(self._list, math.isqrt(a + 2 * block_size + 1))
            ]:
                for t in range((-(a + 1 + p) // 2) % p, block_size, p):
                    block[t] = False
            for idx, p in enumerate(block):
                if p:
                    yield a + 2 * idx + 1
            a += 2 * block_size

    def extend_to_no(self, i):
        """Extend to include the ith prime number.

        Parameters
        ==========

        i : integer

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend_to_no(9)
        >>> sieve._list
        array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])

        Notes
        =====

        The list is extended by 50% if it is too short, so it is
        likely that it will be longer than requested.
        """
        i = int(i)
        while len(self._list) < i:
            self.extend(int(self._list[-1] * 1.5))

    def primerange(self, a, b=None):
        """Generate all prime numbers in the range [2, a) or [a, b).

        Examples
        ========

        >>> from sympy import sieve, prime

        All primes less than 19:

        >>> print([i for i in sieve.primerange(19)])
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> print([i for i in sieve.primerange(7, 19)])
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(sieve.primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        """
        if b is None:
            b = math.ceil(a)
        else:
            a = max(2, math.ceil(a))
            b = math.ceil(b)
        if a >= b:
            return
        self.extend(b)
        yield from self._list[bisect_left(self._list, a) : bisect_left(self._list, b)]

    def totientrange(self, a, b):
        """Generate all totient numbers for the range [a, b).

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.totientrange(7, 18)])
        [6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16]
        """
        a = max(1, math.ceil(a))
        b = math.ceil(b)
        n = len(self._tlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._tlist[i]
        else:
            self._tlist += array("L", range(n, b))
            for i in range(1, n):
                ti = self._tlist[i]
                if ti == i - 1:
                    startindex = (n + i - 1) // i * i
                    for j in range(startindex, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield ti

            for i in range(n, b):
                ti = self._tlist[i]
                if ti == i:
                    for j in range(i, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield self._tlist[i]

    def mobiusrange(self, a, b):
        """Generate all mobius numbers for the range [a, b).

        Parameters
        ==========

        a : integer
            First number in range

        b : integer
            First number outside of range

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.mobiusrange(7, 18)])
        [-1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1]
        """
        a = max(1, math.ceil(a))
        b = math.ceil(b)
        n = len(self._mlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._mlist[i]
        else:
            self._mlist += array("i", [0] * (b - n))
            for i in range(1, n):
                mi = self._mlist[i]
                startindex = (n + i - 1) // i * i
                for j in range(startindex, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

            for i in range(n, b):
                mi = self._mlist[i]
                for j in range(2 * i, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

    def search(self, n):
        """Return the indices i, j of the primes that bound n.

        If n is prime then i == j.

        Although n can be an expression, if ceiling cannot convert
        it to an integer then an n error will be raised.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve.search(25)
        (9, 10)
        >>> sieve.search(23)
        (9, 9)
        """
        test = math.ceil(n)
        n = int(n)
        if n < 2:
            raise ValueError("n should be >= 2 but got: %s" % n)
        if n > self._list[-1]:
            self.extend(n)
        b = bisect(self._list, n)
        if self._list[b - 1] == test:
            return b, b
        else:
            return b, b + 1

    def __contains__(self, n):
        try:
            n = int(n)
            assert n >= 2
        except (ValueError, AssertionError):
            return False
        if n % 2 == 0:
            return n == 2
        a, b = self.search(n)
        return a == b

    def __iter__(self):
        for n in it.count(1):
            yield self[n]

    def __getitem__(self, n):
        """Return the nth prime number"""
        if isinstance(n, slice):
            self.extend_to_no(n.stop)
            start = n.start if n.start is not None else 0
            if start < 1:
                # sieve[:5] would be empty (starting at -1), let's
                # just be explicit and raise.
                raise IndexError("Sieve indices start at 1.")
            return self._list[start - 1 : n.stop - 1 : n.step]
        else:
            if n < 1:
                # offset is one, so forbid explicit access to sieve[0]
                # (would surprisingly return the last one).
                raise IndexError("Sieve indices start at 1.")
            n = int(n)
            self.extend_to_no(n)
            return self._list[n - 1]


# Generate a global object for repeated use in trial division etc
sieve = Sieve()


def primepi(n: int) -> int:
    r"""Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Explanation
    ===========

    In sieve method, we remove all multiples of prime p
    except p itself.

    Let phi(i,j) be the number of integers 2 <= k <= i
    which remain after sieving from primes less than
    or equal to j.
    Clearly, pi(n) = phi(n, sqrt(n))

    If j is not a prime,
    phi(i,j) = phi(i, j - 1)

    if j is a prime,
    We remove all numbers(except j) whose
    smallest prime factor is j.

    Let $x= j \times a$ be such a number, where $2 \le a \le i / j$
    Now, after sieving from primes $\le j - 1$,
    a must remain
    (because x, and hence a has no prime factor $\le j - 1$)
    Clearly, there are phi(i / j, j - 1) such a
    which remain on sieving from primes $\le j - 1$

    Now, if a is a prime less than equal to j - 1,
    $x= j \times a$ has smallest prime factor = a, and
    has already been removed(by sieving from a).
    So, we do not need to remove it again.
    (Note: there will be pi(j - 1) such x)

    Thus, number of x, that will be removed are:
    phi(i / j, j - 1) - phi(j - 1, j - 1)
    (Note that pi(j - 1) = phi(j - 1, j - 1))

    $\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

    So,following recursion is used and implemented as dp:

    phi(a, b) = phi(a, b - 1), if b is not a prime
    phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

    Clearly a is always of the form floor(n / k),
    which can take at most $2\sqrt{n}$ values.
    Two arrays arr1,arr2 are maintained
    arr1[i] = phi(i, j),
    arr2[i] = phi(n // i, j)

    Finally the answer is arr2[1]

    Parameters
    ==========

    n : int

    """
    if n < 2:
        return 0
    if n <= sieve._list[-1]:
        return sieve.search(n)[0]
    lim = math.isqrt(n)
    arr1 = [(i + 1) >> 1 for i in range(lim + 1)]
    arr2 = [0] + [(n // i + 1) >> 1 for i in range(1, lim + 1)]
    skip = [False] * (lim + 1)
    for i in range(3, lim + 1, 2):
        # Presently, arr1[k]=phi(k,i - 1),
        # arr2[k] = phi(n // k,i - 1) # not all k's do this
        if skip[i]:
            # skip if i is a composite number
            continue
        p = arr1[i - 1]
        for j in range(i, lim + 1, i):
            skip[j] = True
        # update arr2
        # phi(n/j, i) = phi(n/j, i-1) - phi(n/(i*j), i-1) + phi(i-1, i-1)
        for j in range(1, min(n // (i * i), lim) + 1, 2):
            # No need for arr2[j] in j such that skip[j] is True to
            # compute the final required arr2[1].
            if skip[j]:
                continue
            st = i * j
            if st <= lim:
                arr2[j] -= arr2[st] - p
            else:
                arr2[j] -= arr1[n // st] - p
        # update arr1
        # phi(j, i) = phi(j, i-1) - phi(j/i, i-1) + phi(i-1, i-1)
        # where the range below i**2 is fixed and
        # does not need to be calculated.
        for j in range(lim, min(lim, i * i - 1), -1):
            arr1[j] -= arr1[j // i] - p
    return arr2[1]
