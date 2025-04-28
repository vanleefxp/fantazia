from __future__ import annotations

from abc import ABCMeta
from typing import overload, Callable, Any
import typing as t
import sys
from functools import partial
from threading import RLock

__all__ = [
    "cachedProp",
    "cachedGetter",
    "ClassPropMeta",
    "classProp",
    "cachedClassProp",
    "cachedGetter",
    "lazyIsInstance",
    "getMethodClass",
    "singleton",
]

_DUMMY = object()

type FGet[T, P] = Callable[[T], P]
type FSet[T, P] = Callable[[T, P], None]
type FDel[T] = Callable[[T], None]
type PropConstructor[T, P, S: property] = Callable[
    [FGet[T, P], FSet[T, P] | None, FDel[T] | None, str | None], type[S]
]

if t.TYPE_CHECKING:  # pragma: no cover

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
    def cachedClassProp[T, P](
        fget: FGet[T, P],
        fset: FSet[T, P] | None = None,
        fdel: FDel[T] | None = None,
        doc: str | None = None,
    ) -> _cachedClassProp[T, P]: ...

    @overload
    def cachedClassProp[T, P](
        key: str,
    ) -> PropConstructor[T, P, _cachedClassProp[T, P]]: ...

    @overload
    def cachedGetter[T, P](fget: FGet[T, P]) -> FGet[T, P]: ...

    @overload
    def cachedGetter[T, P](key: str) -> Callable[[FGet[T, P]], FGet[T, P]]: ...

    @overload
    def singleton[T](cls: type[T]) -> type[T]: ...

    @overload
    def singleton[T](key: str) -> Callable[[type[T]], type[T]]: ...


class _cachedProp[T, P](property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None, *, key: str = None):
        super().__init__(fget, fset, fdel, doc)
        if key is None:
            key = "_" + fget.__name__
        self._key = key

    def __get__(self, instance: T, owner: type[T] = None) -> P:
        with RLock():
            if (value := getattr(instance, self._key, _DUMMY)) is _DUMMY:
                value = super().__get__(instance, owner)
                setattr(instance, self._key, value)
            return value


class ClassPropMeta(ABCMeta):
    def __setattr__(cls, name: str, value: Any):
        if isinstance(desc := vars(cls).get(name), classProp):
            desc.__set__(cls, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(cls, name: str):
        if isinstance(desc := vars(cls).get(name), classProp):
            desc.__delete__(cls)
        else:
            super().__delattr__(name)


class classProp[T, P](property):
    """
    A class-level property that can be overridden by subclasses. Class properties can be
    accessed from both the class and its instances.

    Such thing like a class property used to be possible by chaining the `@property` and
    `@classmethod` decorators, but this approach has become [deprecated since Python 3.11 and
    removed in 3.13](https://docs.python.org/3.13/library/functions.html#classmethod). However,
    in some occasions, it is still useful to have a class-level property that can be extended
    and overridden to convey instance-irrelevant information about the class. This `classProp`
    descriptor allows for the definition of such properties.

    For a class to have class properties, its metaclass must be set to `ClassPropMeta` (or its
    subtype) so that illegal modifications to class properties can be detected.
    """

    def __init__(
        self,
        fget: FGet[T, P] | None = None,
        fset: FSet[T, P] | None = None,
        fdel: FDel[T] | None = None,
        doc: str | None = None,
    ):
        super().__init__(fget, fset, fdel, doc)

    def __get__(self, instance: T | None, owner: type[T] = None) -> P:
        if owner is None:
            owner = instance.__class__

        # The following line is crucial for the program to work.
        # If removed, `NotImplementedError` from abstract class properties will be triggered
        # unexpectedly.
        # From current analysis, the problem is related to the `_abc__abc_init` function in
        # CPython's `_abc.c` module.
        #
        # Python version: 3.13.3
        getattr(owner, "__abstractmethods__")

        return self.fget(owner)

    def __set_name__(self, owner: type[T], name: str):
        if not isinstance(owner, ClassPropMeta):
            raise TypeError(
                f"Class {owner.__name__} must use {ClassPropMeta.__name__} (or its subtype) as "
                "metaclass to have class properties."
            )
        super().__set_name__(owner, name)
        self._cls = owner

    def __set__(self, instance: T, value: P):
        return super().__set__(self._cls, value)

    def __delete__(self, instance: T):
        return super().__delete__(self._cls)


class _cachedClassProp[T, P](classProp[T, P], _cachedProp[T, P]):
    def __init__(
        self,
        fget: FGet[T, P] | None = None,
        fset: FSet[T, P] | None = None,
        fdel: FDel[T] | None = None,
        doc: str | None = None,
        key: str | None = None,
    ):
        super().__init__(fget, fset, fdel, doc)
        if key is None:
            key = "_" + fget.__name__
        self._key = key


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
        return partial(_cachedProp, key=key)
    else:
        return _cachedProp(arg1, *args, **kwargs)


def cachedClassProp(arg1=None, *args, **kwargs):
    hasKey = False
    if isinstance(arg1, str):
        key: str = arg1
        hasKey = True
    elif "key" in kwargs:
        key: str = kwargs.pop("key")
        hasKey = True
    if hasKey:
        return partial(_cachedClassProp, key=key)
    else:
        return _cachedClassProp(arg1, *args, **kwargs)


def _cachedGetter[T, P](fget: FGet[T, P], *, key: str = None) -> FGet[T, P]:
    if key is None:
        fname = fget.__name__
        if fname.startswith("__") and fname.endswith("__"):
            # "dunder" method
            key = f"_{fname[2:-2]}"
        else:
            key = f"_{fname}"

    def wrapper(self: T) -> P:
        with RLock():
            if (value := getattr(self, key, _DUMMY)) is _DUMMY:
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
        return partial(_cachedGetter, key=key)
    else:
        return _cachedGetter(arg1, *args, **kwargs)


def lazyIsInstance(obj: Any, clsName: str) -> bool:
    """
    Judges whether `obj` is an instance of a class with the given name. The target class
    doesn't need to be imported.
    """
    if "." in clsName:
        moduleName, clsName = clsName.rsplit(".", 1)
    else:
        moduleName = "__main__"
    if (module := sys.modules.get(moduleName, _DUMMY)) is not _DUMMY:
        cls = getattr(module, clsName)
        return isinstance(obj, cls)
    else:
        # `obj` cannot be an instance of a class defined in a module that has not been imported,
        # If so, how is `obj` created?
        return False
        # return (
        #     obj.__class__.__name__ == clsName and obj.__class__.__module__ == moduleName
        # )


def getMethodClass(meth: Callable) -> type | None:
    """
    Returns the class where the method `meth` is defined, or `None` if it is not a method of a
    class.
    """
    if hasattr(meth, "__self__"):
        return meth.__self__.__class__
    else:
        if "." not in meth.__qualname__:
            # not a method of a class
            return None
        clsName = meth.__qualname__.split(".")[0]
        modName = meth.__module__
        cls = getattr(sys.modules[modName], clsName)
        return cls


def _singleton[T](cls: type[T], *, key: str = "_instance") -> type[T]:
    origNew = cls.__new__

    def __new__(cls, *args, **kwargs):
        with RLock():
            if (instance := getattr(cls, key, _DUMMY)) is _DUMMY:
                instance = origNew(cls, *args, **kwargs)
                setattr(cls, key, instance)
            return instance

    cls.__new__ = __new__
    return cls


def singleton(arg1=None, *, key=_DUMMY):
    hasKey = False
    if isinstance(arg1, str):
        key = arg1
        hasKey = True
    elif key is not _DUMMY:
        hasKey = True
    if hasKey:
        return partial(_singleton, key=key)
    else:
        return _singleton(arg1)
