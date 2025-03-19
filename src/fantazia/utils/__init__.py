from collections.abc import Iterable
from typing import Sequence, TypeVar, Callable, Type
from bisect import bisect_left
from enum import StrEnum, IntEnum

import numpy as np

__all__ = ["bisect_round", "bsearch", "RoundMode"]

T = TypeVar("T")


class classproperty(property):
    def __get__(self, owner_self: T, owner_cls: Type[T]):
        return self.fget(owner_cls)


class classconst(classproperty):
    def __set__(self, instance, value):
        raise ValueError("Cannot set constant value")


class Rounding(IntEnum):
    FLOOR = -1
    ROUND = 0
    CEIL = 1


class RoundMode(StrEnum):
    HALF_UP = "half-up"
    HALF_EVEN = "half-even"
    HALF_DOWN = "half-down"


def rounding(
    x: float,
    step: float = 1,
    rounding: Rounding = Rounding.ROUND,
    roundMode: RoundMode = RoundMode.HALF_EVEN,
) -> tuple[float, float]:
    q, r = divmod(x, step)
    if r == 0:
        return x, 0
    match rounding:
        case Rounding.FLOOR:
            return q * step, r
        case Rounding.CEIL:
            return (q + 1) * step, r - step
        case Rounding.ROUND:
            if r <= step / 2:
                return q * step, r
            elif r >= step / 2:
                return (q + 1) * step, r - step
            else:
                match roundMode:
                    case RoundMode.HALF_UP:
                        return (q + 1) * step, r - step
                    case RoundMode.HALF_EVEN:
                        return (
                            ((q + 1) * step, r - step) if q % 2 == 0 else (q * step, r)
                        )
                    case RoundMode.HALF_DOWN:
                        return q * step, r
                    case _:
                        raise ValueError(f"Invalid round mode: {roundMode}")
        case _:
            raise ValueError(f"Invalid rounding: {rounding}")


def bisect_round(
    a: Sequence[T],
    x: T,
    lo: int | None = 0,
    hi: int | None = None,
    /,
    key: Callable[[T], int] | None = None,
    roundingMode: RoundMode = RoundMode.HALF_UP,
) -> int:
    """
    Returns the index of item closest to `x` in sequence `a`.

    * If `x` is already in `a`, returns its index.
    * If `x` is less than the first item in `a`, returns `0`.
    * If `x` is greater than the last item in `a`, returns `len(a) - 1`.
    * If `x` is between two items in `a`, returns the index of the item closest to `x`.
      When `x` has the same distance to both items, follows `roundingMode`.
    """
    idx = bisect_left(a, x, lo, hi, key=key)
    if idx >= len(a):
        return idx - 1
    if idx <= 0 or a[idx] == x:
        return idx
    prevItem = a[idx - 1]
    nextItem = a[idx]
    dl = x - prevItem
    dr = nextItem - x
    if dr < dl:
        return idx
    elif dr > dl:
        return idx - 1
    else:
        if roundingMode == RoundMode.HALF_DOWN:
            return idx - 1
        elif roundingMode == RoundMode.HALF_EVEN:
            return idx - 1 if idx % 2 == 0 else idx
        else:
            return idx


def bsearch(
    a: Sequence[T],
    x: T,
    lo: int | None = 0,
    hi: int | None = None,
    key: Callable[[T], int] | None = None,
) -> int:
    idx = bisect_left(a, x, lo, hi, key)
    item = a[idx]
    if item == x:
        return idx
    else:
        return -idx - 1


def _clusterCenters(data: np.ndarray, labels):
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
