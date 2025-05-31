from functools import cache, lru_cache
from numbers import Integral
import math

from .abc.wrapper import PitchWrapperBase, OPitchWrapper, PitchWrapper
from .abc.edo import EDOPitchBase, OEDOPitch, EDOPitch
from .edo12 import edo12, OPitch, Pitch
from ..utils.cls import classProp

__all__ = ["edo", "edo12", "edo17", "edo19", "edo22", "edo31", "edo41", "edo53"]

LOG_2_3_M1 = math.log2(1.5)


@lru_cache
def edo(n: Integral) -> type[EDOPitchBase]:
    n: int = int(n)
    fifthSize = round(n * LOG_2_3_M1)
    d = math.gcd(n, fifthSize)
    n //= d
    fifthSize //= d

    if n == 12:
        return edo12
    pitchBaseType = _createEdo(n)
    pitchBaseType._fifthSize = fifthSize
    return pitchBaseType


@cache
def _createEdo(n: int) -> type[EDOPitchBase]:
    class edo_n(
        PitchWrapperBase["_OPitch", "_Pitch"], EDOPitchBase["_OPitch", "_Pitch"]
    ):
        @classProp
        def edo(cls) -> int:
            return n

        @classProp
        def OPitch(self) -> type["_OPitch"]:
            return _OPitch

        @classProp
        def Pitch(self) -> type["_Pitch"]:
            return _Pitch

        def __str__(self):
            return f"{self._p!s}@edo{self.edo}"

    class _OPitch(edo_n, OPitchWrapper["_Pitch"], OEDOPitch["_Pitch"]):
        __slots__ = ("_p", "_hash")

        def __new__(cls, *args, **kwargs):
            return cls._newImpl(OPitch(*args, **kwargs))

    class _Pitch(edo_n, PitchWrapper["_OPitch"], EDOPitch["_OPitch"]):
        __slots__ = ("_p", "_hash", "_opitch")

        def __new__(cls, *args, **kwargs):
            return cls._newImpl(Pitch(*args, **kwargs))

    edo_n.__name__ = f"edo{n}"
    _OPitch.__name__ = f"OPitch{n}"
    _Pitch.__name__ = f"Pitch{n}"

    return edo_n


edo17 = edo(17)
edo19 = edo(19)
edo22 = edo(22)
edo31 = edo(31)
edo41 = edo(41)
edo53 = edo(53)
