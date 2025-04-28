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
from abc import ABCMeta
import typing as t
from bisect import bisect_left

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


class OrderedSet[T](Sequence[T], Set[T], metaclass=ABCMeta):
    def index(self, value: T, start: int = 0, stop: int = ...) -> int:
        if stop is ...:
            stop = None
        idx = bisect_left(self, value, start, stop)
        if idx == len(self) or self[idx] != value:
            raise ValueError(f"{value} is not in set")
        return idx
