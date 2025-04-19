from typing import Type, Never

_NOT_FOUND = object()


class classproperty[InstanceT, PropT](property):
    def __get__(self, instance: InstanceT, owner: Type[InstanceT] = None) -> PropT:
        return self.fget(owner)


class classconst[InstanceT, PropT](classproperty):
    __slots__ = ("_name",)

    def __set_name__(self, owner: Type[InstanceT], name: str):
        self._name = name

    def __get__(self, instance: InstanceT, owner: Type[InstanceT] = None) -> PropT:
        if hasattr(owner, f"_{self._name}"):
            value = getattr(owner, f"_{self._name}")
        else:
            value = super().__get__(instance, owner)
            setattr(owner, f"_{self._name}", value)
        return value

    def __set__(self, instance: InstanceT, value: PropT) -> Never:
        raise ValueError("Cannot set constant value.")
