from abc import ABCMeta
import typing as t

if t.TYPE_CHECKING:  # pragma: no cover
    import music21 as m21


class M21Mixin[M21Type: m21.prebase.ProtoM21Object](metaclass=ABCMeta):
    def m21(self, *args, **kwargs) -> M21Type:
        raise NotImplementedError
