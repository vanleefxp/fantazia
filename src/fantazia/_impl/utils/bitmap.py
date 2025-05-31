import itertools as it
from collections.abc import Sequence, MutableSequence, Iterable, Iterator, Buffer
from numbers import Integral
from typing import Self, overload, Any
import typing as t
from functools import lru_cache, reduce
from abc import ABCMeta, abstractmethod
import operator as op

_MISSING = object()
_BITMASK = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80)


def _setBit(ba: MutableSequence[int], pos: int) -> bytearray:
    i_byte, i_bit = divmod(pos, 8)
    ba[i_byte] |= _BITMASK[i_bit]
    return ba


def _unsetBit(ba: MutableSequence[int], pos: int) -> bytearray:
    i_byte, i_bit = divmod(pos, 8)
    ba[i_byte] &= ~_BITMASK[i_bit]
    return ba


def _setBitValue(ba: MutableSequence[int], pos: int, value: bool) -> bytearray:
    i_byte, i_bit = divmod(pos, 8)
    if value:
        ba[i_byte] |= _BITMASK[i_bit]
    else:
        ba[i_byte] &= ~_BITMASK[i_bit]
    return ba


def _getBitValue(ba: Buffer, pos: int) -> bool:
    i_byte, i_bit = divmod(pos, 8)
    return (ba[i_byte] & _BITMASK[i_bit]) != 0


def _bools2byte(booleans: Iterable[bool]) -> int:
    val = 0
    for i, bit in enumerate(booleans):
        if bit:
            val |= _BITMASK[i]
    return val


def _bytes_and(a: Sequence[int], b: Sequence[int]) -> bytes:
    if len(b) >= len(a):
        itr = zip(a, b)
    else:
        itr = it.zip_longest(a, b, fillvalue=0)
    return bytes(x & y for x, y in itr)


def _bytes_or(a: Sequence[int], b: Sequence[int]) -> bytes:
    if len(b) >= len(a):
        itr = zip(a, b)
    else:
        itr = it.zip_longest(a, b, fillvalue=255)
    return bytes(x | y for x, y in itr)


def _bytes_xor(a: Sequence[int], b: Sequence[int]) -> bytes:
    if len(b) >= len(a):
        itr = zip(a, b)
    else:
        itr = it.zip_longest(a, b, fillvalue=0)
    return bytes(x ^ y for x, y in itr)


def _bytes_not(b: bytes) -> bytes:
    return bytes(~x & 0xFF for x in b)


def _bytes_regularize(b: bytes, length: int) -> bytes:
    """
    Process a `bytes` sequence to match the desired bitmap length.
    """
    n_bytes, n_bits = divmod(length, 8)
    if n_bits == 0:
        if len(b) > n_bytes:
            return b[:n_bytes]
        elif len(b) == n_bytes:
            return b
        else:
            return b + b"\x00" * (n_bytes - len(b))
    else:
        if len(b) >= n_bytes + 1:
            return b[:n_bytes] + bytes((b[n_bytes] & (0xFF >> (8 - n_bits)),))
        else:
            return b + b"\x00" * (n_bytes + 1 - len(b))


# def _bytes_or(a: Sequence[int], b: Sequence[int]) -> bytes:
#     return bytes(x | y for x, y in _bytes_binop_itr(a, b))


# def _bytes_xor(a: Sequence[int], b: Sequence[int]) -> bytes:
#     return bytes(x ^ y for x, y in _bytes_binop_itr(a, b))


class Bitmap(Sequence[bool], metaclass=ABCMeta):
    __slots__ = ()
    ...


class MutableBitmap(Bitmap, MutableSequence[bool]):
    __slots__ = ()

    @abstractmethod
    def set(self, idx) -> Self:
        raise NotImplementedError

    @abstractmethod
    def unset(self, idx) -> Self:
        raise NotImplementedError


class frozenbmap(Bitmap):
    __slots__ = ("_length", "_data")

    _length: int
    _data: int

    if t.TYPE_CHECKING:

        @overload
        def __new__(
            cls,
            data: str | Iterable[bool] | Buffer | None = None,
            /,
            *,
            length: Integral | None = None,
        ) -> Self: ...

    def __new__(
        cls,
        arg0=_MISSING,
        /,
        *,
        length=_MISSING,
        byteorder="little",
        bufferLike=True,
    ) -> Self:
        if arg0 is _MISSING:
            if length == 0:
                return _EMPTY_BITMAP
            return cls._newHelper(0, int(length))
        if isinstance(arg0, cls):
            if length is _MISSING or length == len(arg0):
                return arg0
            length = int(length)
            data = arg0._data
            if arg0._length > length:
                data = (~((-1) << length)) & data
            return cls._newImpl(data, length)
        if isinstance(arg0, Buffer) and bufferLike:
            data = int.from_bytes(arg0, byteorder=byteorder)
            if length is _MISSING:
                length = len(arg0) * 8
            else:
                length = int(length)
                data = (~((-1) << length)) & data
            return cls._newImpl(data, length)
        if isinstance(arg0, str):
            # from a string with zeroes and ones
            data = int(arg0[::-1], base=2)
            if length is _MISSING:
                length = len(arg0)
            else:
                length = int(length)
                data = (~((-1) << length)) & data
            return cls._newImpl(data, length)
        if isinstance(arg0, Iterable):
            # from an iterable containing `bool`s
            if length is _MISSING:
                if not isinstance(arg0, Sequence):
                    arg0 = tuple(arg0)
                length = len(arg0)
                data: int = reduce(
                    op.or_, (((1 << i) if bit else 0 for i, bit in enumerate(arg0)))
                )
            else:
                length = int(length)
                data: int = reduce(
                    op.or_,
                    (
                        (1 << i) if bit else 0
                        for i, bit in enumerate(it.islice(arg0, length))
                    ),
                )
            return cls._newImpl(data, length)

    @classmethod
    @lru_cache
    def _newHelper(cls, data: int, length: int) -> Self:
        return cls._newImpl(data, length)

    @classmethod
    def _newImpl(cls, data: int, length: int) -> Self:
        self = super().__new__(cls)
        self._length = length
        self._data = data
        return self

    def __len__(self) -> int:
        return self._length

    def __bytes__(self) -> bytes:
        return self._data.to_bytes((self._length + 7) // 8, byteorder="little")

    def __int__(self) -> int:
        return self._data

    def __iter__(self) -> Iterator[bool]:
        for i in range(self._length):
            yield self._data & (1 << i) != 0

    def __getitem__(self, idx: Integral | slice | Iterable[Integral]) -> bool | Self:
        if isinstance(idx, Integral):
            return self._getitem_idx(idx)
        elif isinstance(idx, slice):
            return self._getitem_slice(idx)
        elif isinstance(idx, Iterable):
            return self._getitem_multiIdx(idx)
        else:
            raise TypeError("unsupported index type")

    def _getitem_idx(self, idx: Integral) -> bool:
        return self._data & (1 << idx) != 0

    def _extract(self, seq: Sequence[Integral]) -> Self:
        newLength = len(seq)
        newData = 0
        for i, j in enumerate(seq):
            newData |= (1 << i) if self._data & (1 << j) != 0 else 0
        return self._newHelper(newData, newLength)

    def _getitem_slice(self, idx: slice) -> Self:
        start, stop, step = idx.indices(self._length)
        r = range(start, stop, step)
        return self._extract(r)

    def _getitem_multiIdx(self, idx: Iterable[Integral]) -> Self:
        idx = tuple(idx)
        return self._extract(idx)

    def __str__(self) -> str:
        return f"{self._data:0{self._length}b}"[::-1]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self!s}")'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, frozenbmap):
            return self._data == other._data and self._length == other._length
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash((self._data, self._length))

    def _resolveOperand(self, other: Any) -> int | Any:
        if isinstance(other, self.__class__):
            other = other._data
        elif isinstance(other, Buffer):
            other = int.from_bytes(other, byteorder="little")
        return other

    def __and__(self, other: Any) -> Self:
        other = self._resolveOperand(other)
        newData = (self._data & other) & (~((-1) << self._length))
        return self._newImpl(newData, self._length)

    def __or__(self, other: Any) -> Self:
        other = self._resolveOperand(other)
        newData = (self._data | other) & (~((-1) << self._length))
        return self._newImpl(newData, self._length)

    def __xor__(self, other: Any) -> Self:
        other = self._resolveOperand(other)
        newData = (self._data ^ other) & (~((-1) << self._length))
        return self._newImpl(newData, self._length)

    def __invert__(self) -> Self:
        newData = (~self._data) & (~((-1) << self._length))
        return self._newImpl(newData, self._length)

    def __lshift__(self, other: Integral) -> Self:
        newData = (self._data << other) & (~((-1) << self._length))
        return self._newImpl(newData, self._length)

    def __rshift__(self, other: Integral) -> Self:
        newData = (self._data >> other) & (~((-1) << self._length))
        return self._newImpl(newData, self._length)

    def __le__(self, other: Any) -> bool:
        other = self._resolveOperand(other)
        return self._data <= other

    def __lt__(self, other: Any) -> bool:
        other = self._resolveOperand(other)
        return self._data < other

    def roll(self, shift: Integral) -> Self:
        shift = shift % self._length
        newData = (self._data << shift) & (~((-1) << self._length)) | (
            self._data >> (self._length - shift)
        )
        return self._newImpl(newData, self._length)


_EMPTY_BITMAP = frozenbmap._newImpl(0, 0)


if __name__ == "__main__":
    import sys

    bm1 = frozenbmap((True, True, True, False, True, True, False, True), length=7)
    bm2 = frozenbmap("0100001")
    print(bm1, bm2, bm1[0])
    print(bm1 & bm2, bm1 | bm2, bm1 ^ bm2, ~bm1)
    print(bm1 <= bm2, bm1 < bm2)
    print(sys.getsizeof(bm1), sys.getsizeof(bm2))
