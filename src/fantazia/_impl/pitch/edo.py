from functools import cache, lru_cache
from numbers import Integral
import math

from .abc import edo as _abc_edo, wrapper as _abc_wrapper
from . import edo12 as _edo12
from .edo12 import Notation as edo12
from ..utils.cls import classProp

__all__ = ["edo", "edo12", "edo17", "edo19", "edo22", "edo31", "edo41", "edo53"]

LOG_2_3_M1 = math.log2(1.5)


@lru_cache
def edo(n: Integral) -> type[_abc_edo.Notation]:
    n: int = int(n)
    fifthSize = round(n * LOG_2_3_M1)
    d = math.gcd(n, fifthSize)
    n //= d
    fifthSize //= d

    if n == 12:
        return _edo12.Notation
    pitchBaseType = _createEdo(n)
    pitchBaseType._fifthSize = fifthSize
    return pitchBaseType


@cache
def _createEdo(n: int) -> type[_abc_edo.Notation]:
    class _Notation(
        _abc_wrapper.Notation["_OPitch", "_Pitch"],
        _abc_edo.Notation["_OPitch", "_Pitch"],
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

    class _OPitch(_Notation, _abc_wrapper.OPitch["_Pitch"], edo.OEDOPitch["_Pitch"]):
        __slots__ = ("_p", "_hash")

        def __new__(cls, *args, **kwargs):
            return cls._newImpl(_edo12.OPitch(*args, **kwargs))

    class _Pitch(_Notation, _abc_wrapper.Pitch["_OPitch"], edo.EDOPitch["_OPitch"]):
        __slots__ = ("_p", "_hash", "_opitch")

        def __new__(cls, *args, **kwargs):
            return cls._newImpl(_edo12.Pitch(*args, **kwargs))

    return _Notation


edo17 = edo(17)
edo19 = edo(19)
edo22 = edo(22)
edo31 = edo(31)
edo41 = edo(41)
edo53 = edo(53)
