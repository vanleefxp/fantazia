from __future__ import annotations

from collections.abc import (
    Mapping,
    MutableMapping,
    Set,
    MutableSet,
    Sequence,
    MutableSequence,
    Iterable,
)
from collections import Counter
import typing as t
from numbers import Integral, Real

type SliceInput = slice[Integral, Integral, Integral]
type Slice = slice[int, int, int]

if t.TYPE_CHECKING:
    from typing import overload, Protocol
    from _typeshed import SupportsAdd, SupportsMul

    class SupportsAddAndMul(SupportsAdd, SupportsMul, Protocol): ...

    @overload
    def updated[K, V](
        dst: MutableMapping[K, V], *srcs: Mapping[K, V]
    ) -> MutableMapping[K, V]: ...

    @overload
    def updated[T](dst: MutableSet[T], *srcs: Set[T]) -> MutableSet[T]: ...

    @overload
    def updated[T](
        dst: MutableSequence[T], *srcs: Iterable[T]
    ) -> MutableSequence[T]: ...

    @overload
    def cycGet[T](seq: Sequence[T], idx: int) -> T: ...

    @overload
    def cycGet[T: SupportsAddAndMul](seq: Sequence[T], idx: int, increment: T) -> T: ...


def updated(dst, *srcs):
    if isinstance(dst, MutableSequence):
        for src in srcs:
            dst.extend(src)
        return dst
    for src in srcs:
        dst.update(src)
    return dst


def cycGet(seq, idx, increment=None):
    q, r = divmod(idx, len(seq))
    res = seq[r]
    if increment is not None:
        res += increment * q
    return res


counter_update = Counter.update
counter_subtract = Counter.subtract


def counter_cleanZeroes[T, N: Real](m: MutableMapping[T, N]) -> None:
    for k in frozenset(k for k, v in m.items() if v == 0):
        del m[k]


# class OrderedSet[T](Sequence[T], Set[T], metaclass=ABCMeta):
#     def index(self, value: T, start: int = 0, stop: int = ...) -> int:
#         if stop is ...:
#             stop = None
#         idx = bisect_left(self, value, start, stop)
#         if idx == len(self) or self[idx] != value:
#             raise ValueError(f"{value} is not in set")
#         return idx


# def sliceExact(
#     sl: SliceInput, length: Integral, *, asSlice: bool = True
# ) -> tuple[int, int, int] | Slice:
#     start, stop, step = sl.indices(length)
#     if step > 0:
#         if stop < start:
#             stop = start
#         else:
#             n = -((stop - start) // (-step))
#             stop = start + n * step
#     elif step < 0:
#         if stop > start:
#             n = 0
#             stop = start
#         else:
#             n = -((start - stop) // step)
#             stop = start + n * step
#             if stop < 0 and asSlice:
#                 stop = None
#     else:
#         raise ValueError("slice step cannot be zero")
#     if asSlice:
#         return slice(start, stop, step)
#     return start, n, step


# def sliceReverse(
#     sl: slice[Integral, Integral, Integral], length: Integral
# ) -> slice[Integral, Integral]:
#     start, stop, step = sliceExact(sl, length)
#     return stop - 1, start - 1, -step


# def slice2Range(sl: SliceInput, length: Integral) -> range:
#     return range(*sl.indices(length))


# class CView[T](Collection[T]):
#     __slots__ = ("_data",)

#     def __new__(cls, data: Collection[T]) -> Self:
#         self = super().__new__(cls)
#         self._data = data
#         return self

#     def __len__(self) -> int:
#         return len(self._data)

#     def __iter__(self) -> Iterator[T]:
#         return iter(self._data)

#     def __contains__(self, value: Any) -> bool:
#         return value in self._data


# class SeqView[T](CView[T], Sequence[T]):
#     _data: Sequence[T]

#     def __getitem__(self, idx: Integral) -> T:
#         return self._data[idx]

#     def __reversed__(self) -> Iterator[T]:
#         return reversed(self._data)

#     def index(self, value: T, start: int = 0, stop: int = ...) -> int:
#         return self._data.index(value, start, stop)

#     def count(self, value: T) -> int:
#         return self._data.count(value)


# @singleton
# class EmptyView[T](Sequence[T], Set[T]):
#     def __new__(cls, data):
#         return super().__new__(data)

#     def __len__(self) -> int:
#         return 0

#     def __iter__(self) -> Iterator[T]:
#         return iter(())

#     def __reversed__(self):
#         return iter(())

#     def __contains__(self, value: Any) -> bool:
#         return False

#     def __getitem__(self, idx: Integral) -> Never:
#         raise IndexError("index out of range")

#     def index(self, value: Any, start: Integral = 0, stop: Integral = ...):
#         raise ValueError(f"{value} is not in set")


# class RepeatView[T](Sequence[T]):
#     _value: T
#     _count: int

#     def __new__(cls, value: T, count: int):
#         if count < 0:
#             raise ValueError("count must be non-negative")
#         return cls._newHelper(value, int(count))

#     @classmethod
#     def _newHelper(cls, value: T, count: int):
#         self = super().__new__(cls)
#         self._value = value
#         self._count = count
#         return self

#     def __getitem__(self, idx: int) -> T:
#         # TODO)) add slicing support
#         if idx < -self._count or idx >= self._count:
#             raise IndexError("index out of range")
#         return self._value

#     def __len__(self) -> int:
#         return self._count

#     def __contains__(self, value: Any) -> bool:
#         return value == self._value

#     def __iter__(self) -> Iterator[T]:
#         return it.repeat(self._value, self._count)

#     def __reversed__(self):
#         return iter(self)


# class CycleView[T](Sequence[T]):
#     _data: Sequence[T]
#     _count: int

#     def __new__(cls, data: Sequence[T], count: int) -> Self:
#         self = super().__new__(cls)
#         self._data = data
#         self._count = count
#         return self

#     def __len__(self) -> int:
#         return len(self._data) * self._count

#     def __getitem__(self, idx: int) -> T:
#         if idx < 0 or idx >= len(self):
#             raise IndexError("index out of range")
#         return self._data[idx % len(self._data)]

#     def __contains__(self, value: Any) -> bool:
#         return value in self._data

#     def __iter__(self) -> Iterator[T]:
#         return it.islice(it.cycle(self._data), self._count)

#     def __reversed__(self) -> Iterator[T]:
#         return it.islice(it.cycle(reversed(self._data)), len(self))


# class ConcatView[T](Sequence[T]):
#     _data: tuple[Sequence[T]]

#     def __new__(cls, *data: Sequence[T]) -> Self:
#         self = super().__new__(cls)
#         self._data = data
#         return self

#     def __len__(self) -> int:
#         return sum(map(len, self._data))

#     def __getitem__(self, idx: int) -> T:
#         if idx < 0:
#             idx += len(self)
#         for seq in self._data:
#             if idx < len(seq):
#                 return seq[idx]
#             idx -= len(seq)
#         raise IndexError("index out of range")

#     def __iter__(self):
#         return it.chain.from_iterable(self._data)


# class SliceView[T](Sequence[T]):
#     _data: Sequence[T]
#     _slice: slice[int, int, int]

#     def __new__(cls, data: Sequence[T], sl: slice[int, int, int]):
#         self = super().__new__(cls)
#         self._data = data
#         self._slice = sl
#         return self

#     def _slice2Range(self) -> range:
#         return slice2Range(self._slice, len(self._data))

#     def __len__(self) -> int:
#         return len(self._slice2Range())

#     def __getitem__(self, idx: int) -> T:
#         return self._data[self._slice2Range()[idx]]

#     def __iter__(self) -> Iterator[T]:
#         return map(self._data.__getitem__, self._slice2Range())

#     def __reversed__(self) -> Iterator[T]:
#         return map(
#             self._data.__getitem__,
#             reversed(range(*self._slice.indices(len(self._data)))),
#         )


# def repeated[T](value: T, count: Integral) -> Sequence[T]:
#     """
#     Creates a sequence view of a value repeating `count` times.
#     """
#     if count < 0:
#         raise ValueError("count must be non-negative")
#     if count == 0:
#         return EmptyView[T]()
#     # `count` must be converted to builtin `int` type,
#     # because it can be of any `Integral` type, including `sympy.Integer`, or `numpy` integer
#     # types.
#     # This conversion guarantees consistency.
#     return RepeatView(value, int(count))


# def cycled[T](data: Sequence[T], count: Integral) -> Sequence[T]:
#     """
#     Creates a sequence view of a sequence repeating `count` times,
#     """
#     if count < 0:
#         raise ValueError("count must be non-negative")
#     if count == 0 or len(data) == 0:
#         return EmptyView[T]()
#     if len(data) == 1:
#         # single element sequence converted to `RepeatView`
#         return RepeatView(data[0], int(count))
#     # multi-element sequence converted to `CycleView`
#     return CycleView(data, int(count))


# class CastingView[S, T](Sequence[T]):
#     __slots__ = ("_data", "_dtype")

#     _data: Sequence[S]
#     _dtype: type[T]

#     def __new__(cls, data: Sequence[S], dtype: type[T]):
#         self = super().__new__(cls)
#         self._data = data
#         self._dtype = dtype
#         return self

#     def __len__(self) -> int:
#         return len(self._data)

#     def __getItem__(self, idx: Integral) -> T:
#         return self._dtype(self._data[idx])

#     def __iter__(self) -> Iterator[T]:
#         return map(self._dtype, self._data)
