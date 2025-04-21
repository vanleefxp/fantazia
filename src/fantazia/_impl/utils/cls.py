from __future__ import annotations

from typing import Type, overload, Callable
import typing as t

_NOT_FOUND = object()

type FGet[T, P] = Callable[[T], P]
type FSet[T, P] = Callable[[T, P], None]
type FDel[T] = Callable[[T], None]
type PropConstructor[T, P, S: property] = Callable[
    [FGet[T, P], FSet[T, P] | None, FDel[T] | None, str | None], Type[S]
]

if t.TYPE_CHECKING:

    @overload
    def cachedProp[T, P](
        fget: FGet[T, P],
        fset: FSet[T, P] | None = None,
        fdel: FDel[T] | None = None,
        doc: str | None = None,
    ) -> _cachedProp[T, P]: ...

    @overload
    def cachedProp[T, P](
        key: str,
    ) -> PropConstructor[T, P, _cachedProp[T, P]]: ...

    @overload
    def cachedGetter[T, P](fget: FGet[T, P]) -> FGet[T, P]: ...

    @overload
    def cachedGetter[T, P](key: str) -> Callable[[FGet[T, P]], FGet[T, P]]: ...


class _cachedProp[T, P](property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, *, key: str = None):
        super().__init__(fget, fset, fdel, doc)
        if key is None:
            key = "_" + fget.__name__
        self._key = key

    def __get__(self, instance: T, owner: Type[T] = None) -> P:
        if (value := getattr(instance, self._key, _NOT_FOUND)) is _NOT_FOUND:
            value = super().__get__(instance, owner)
            setattr(instance, self._key, value)
        return value


def cachedProp(arg1=None, *args, **kwargs):
    """
    A cached property decorator that caches the result of the property on the instance.
    The cached value is stored, by default, in a private attribute with the same name as the
    property preceded by a leading underscore, or, when a key is specified, with the key as
    the attribute name. On successive calls, the cached value is returned.

    **Note**: This works differently from the `@cached_property` decorator in `functools`,
    which relies on the `__dict__` attribute of the instance to store cached value and is thus
    not compatible with classes that have `__slots__` defined. Make sure that the desired key
    is included in `__slots__` when using this decorator on a slotted class.
    """
    hasKey = False
    if isinstance(arg1, str):
        key: str = arg1
        hasKey = True
    elif "key" in kwargs:
        key: str = kwargs.pop("key")
        hasKey = True
    if hasKey:
        return lambda fget, fset=None, fdel=None, doc=None: _cachedProp(
            fget, fset, fdel, doc, key=key
        )
    else:
        return _cachedProp(arg1, *args, **kwargs)


def _cachedGetter[T, P](fget: FGet[T, P], *, key: str = None) -> FGet[T, P]:
    if key is None:
        fname = fget.__name__
        if fname.startswith("__") and fname.endswith("__"):
            # "dunder" method
            key = f"_{fname[2:-2]}"
        else:
            key = f"_{fname}"

    def wrapper(self: T) -> P:
        if (value := getattr(self, key, _NOT_FOUND)) is _NOT_FOUND:
            value = fget(self)
            setattr(self, key, value)
        return value

    wrapper.__name__ = fget.__name__
    wrapper.__doc__ = fget.__doc__
    return wrapper


def cachedGetter(arg1=None, *args, **kwargs):
    """
    Similar to `cachedProp`, but for a getter function. The cached value is stored in a
    private attribute with the same name as the getter function preceded by a leading
    underscore, or, when a key is specified, with the key as the attribute name.
    """
    hasKey = False
    if isinstance(arg1, str):
        key = arg1
        hasKey = True
    elif "key" in kwargs:
        key = kwargs.pop("key")
        hasKey = True
    if hasKey:
        return lambda fget: _cachedGetter(fget, key=key)
    else:
        return _cachedGetter(arg1, *args, **kwargs)
