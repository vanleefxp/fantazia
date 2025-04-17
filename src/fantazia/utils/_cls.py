from typing import Type


class classproperty[T](property):
    def __get__(self, owner_self: T, owner_cls: Type[T]):
        return self.fget(owner_cls)


class classconst(classproperty):
    def __set__(self, instance, value):
        raise ValueError("Cannot set constant value")
