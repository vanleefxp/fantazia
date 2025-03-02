from collections.abc import Iterable
from typing import Sequence, TypeVar, Callable
from bisect import bisect_left
from enum import StrEnum

import numpy as np
from sklearn.cluster import DBSCAN

__all__ = ["bisect_round", "bsearch", "RoundingMode"]

T = TypeVar("T")


class RoundingMode(StrEnum):
    HALF_UP = "half-up"
    HALF_EVEN = "half-even"
    HALF_DOWN = "half-down"


def bisect_round(
    a: Sequence[T],
    x: T,
    lo: int | None = 0,
    hi: int | None = None,
    /,
    key: Callable[[T], int] | None = None,
    roundingMode: RoundingMode = RoundingMode.HALF_UP,
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
    dr = nextItem - prevItem
    if dr < dl:
        return idx
    elif dr > dl:
        return idx - 1
    else:
        if roundingMode == RoundingMode.HALF_DOWN:
            return idx - 1
        elif roundingMode == RoundingMode.HALF_EVEN:
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
