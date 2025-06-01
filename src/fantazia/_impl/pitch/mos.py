from __future__ import annotations

from ..math_.mos import IntMOSPattern
from .abc import mos as _mos
from ..utils.cls import classProp


def _createMOS(pattern: IntMOSPattern):
    class MOS(_mos.Notation["_OPitch", "_Pitch"]):
        @classProp
        def OPitch(self) -> type[_OPitch]:
            return _OPitch

        @classProp
        def Pitch(self) -> type[_Pitch]:
            return _Pitch

    class _OPitch(_mos.OPitch["_Pitch"]): ...

    class _Pitch(_mos.Pitch["_OPitch"]): ...

    return MOS
