from collections.abc import Callable
from typing import Self
from numbers import Integral
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from bisect import bisect_left, bisect_right

import numpy as np

from .utils.bisect_utils import bisect_round, RoundingMode

__all__ = [ 
    "Accis", "Degs", "Intervals", "Qualities", "AcciPrefs", 
    "PitchLike", "OPitch", "Pitch" 
]

class Accis:
    """
    Common Accidental constants.
    """
    TRIPLE_SHARP = SSS = 3
    DOUBLE_SHARP = SS = 2
    SHARP = S = 1
    NATURAL = N = 0
    FLAT = F = -1
    DOOUBLE_FLAT = FF = -2
    TRIPLE_FLAT = FFF = -3

class Degs:
    """
    Degree constants. Supports both letter names and fixed-do solfege notations.
    """
    DO = C = 0
    RE = D = 1
    MI = E = 2
    FA = F = 3
    SOL = G = 4
    LA = A = 5
    SI = B = 6

class Intervals:
    """
    Common intervals.
    """
    UNISON = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    FIFTH = 4
    SIXTH = 5
    SEVENTH = 6

class Qualities:
    """
    Common interval qualities.
    """
    DOUBLY_AUGMENTED = 3
    AUGMENTED = 2
    MAJOR = 1
    PERFECT = 0
    MINOR = -1
    DIMINISHED = -2
    DOUBLY_DIMINISHED = -3

# major scale notes in circle of fifths order
_majorScaleCo5Order = np.arange ( -1, 6 ) * 7 % 12
_majorScaleCo5Order.flags.writeable = False

# major scale notes in increasing order
_majorScale = np.sort ( _majorScaleCo5Order )
_majorScale.flags.writeable = False

_degreeMask = np.full ( 12, -1, dtype = int )
_degreeMask [ _majorScale ] = np.arange ( 7 )
_degreeMask.flags.writeable = False

_perfectIntervals = frozenset (( 0, 3, 4 ))
_noteNames = np.roll ( np.array ([ chr ( 65 + i ) for i in range ( 7 ) ]), -2 )

type AcciPref = Callable [ [ float ], int ]
"""
Type alias for a function that takes a tone in half-steps and returns the preferred degree for 
that tone. The accidental can be later computed by taking the difference between the given
tone and the standard reference tone in C major scale.
"""

def _makeClosestAcciPref ( roundingMode: RoundingMode ) -> AcciPref:
    def _pref ( tone: float ) -> int:
        return bisect_round ( _majorScale, tone, roundingMode = roundingMode )
    return _pref

def _alwaysSharp ( tone: float ) -> int:
    return bisect_right ( _majorScale, tone ) - 1

def _alwaysFlat ( tone: float ) -> int:
    return bisect_left ( _majorScale, tone )

class AcciPrefs:
    """See `AcciPref` for details."""
    
    SHARP = _alwaysSharp
    """
    Always use the lower degree and sharp sign when a tone is not a standard tone in C major 
    scale, both for standard 12edo pitches and microtonal pitches.
    
    Examples:
    
    | input `tone` | output `degree` | preferred name
    |-:|-:|:-|
    | `6` | `3` | F sharp |
    | `5.5` | `3` | F quarter-tone-sharp |
    | `6.5` | `3` | F 3-quarter-tones-sharp |
    """
    
    FLAT = _alwaysFlat
    """
    Always use the upper degree and flat sign when a tone is not a standard tone in C major 
    scale,both for standard 12edo pitches and microtonal pitches.
    
    Examples:
    
    | input `tone` | output `degree` | preferred name
    |-:|-:|:-|
    | `6` | `4` | G flat |
    | `5.5` | `4` | G 3-quarter-tones-flat |
    | `6.5` | `4` | G quarter-tone-flat |
    """
    
    CLOSEST_SHARP = _makeClosestAcciPref ( RoundingMode.HALF_DOWN )
    """
    For 12edo pitches, use the lower degree and sharp sign when the tone is not a standard tone 
    in C major scale. For microtonal pitches, choose the closest standard tone in C major scale.
    
    Examples:
    
    | input `tone` | output `degree` | preferred name
    |-:|-:|:-|
    | `6` | `3` | F sharp |
    | `5.5` | `3` | F quarter-tone-sharp |
    | `6.5` | `4` | G quarter-tone-flat |
    """
    
    CLOSEST_FLAT = _makeClosestAcciPref ( RoundingMode.HALF_UP )
    """
    For 12edo pitches, use the upper degree and flat sign when the tone is not a standard tone 
    in C major scale. For microtonal pitches, choose the closest standard tone in C major scale.
    
    Examples:
    
    | input `tone` | output `degree` | preferred name |
    |-:|-:|:-|
    | `6` | `4` | G flat |
    | `5.5` | `3` | F quarter-tone-sharp |
    | `6.5` | `4` | G quarter-tone-flat |
    """
    
class PitchLike ( metaclass = ABCMeta ):
    """Base class for `OPitch` and `Pitch`."""
    
    @property
    @abstractmethod
    def deg ( self ) -> int: raise NotImplementedError
    
    @property
    @abstractmethod
    def acci ( self ) -> float: 
        """Accidental of the pitch, in half-steps."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def tone ( self ) -> float: raise NotImplementedError
    
    @property
    @abstractmethod
    def quality ( self ) -> float:
        """
        Interval quality when regarding the pitch as an interval.
        """
        raise NotImplementedError
    
    @property
    def freq ( self ) -> float:
        """
        Frequency of the pitch relative to middle C.
        """
        return np.pow ( 2, self.tone / 12 )
    
    @abstractmethod
    def __add__ ( self, other: Self ) -> Self:
        raise NotImplementedError
    
    @abstractmethod
    def __neg__ ( self ) -> Self:
        raise NotImplementedError
    
    def __sub__ ( self, other: Self ) -> Self:
        return self + ( -other )
    
    def __lt__ ( self, other: Self ) -> bool:
        return self.tone < other.tone or ( self.tone == other.tone and self.deg < other.deg )
    

@dataclass ( frozen = True, slots = True )
class OPitch ( PitchLike ):
    """
    Represents a pitch in an octave, or equivalently, an interval no greater than an octave.
    """
    
    deg: int = Degs.C
    """Degree of the pitch in C major scale, not considering accidentals."""
    
    acci: float = Accis.NATURAL
    """Accidental of the pitch, in half-steps."""
    
    @classmethod
    def fromDegAndTone ( cls, deg: int, tone: float ) -> Self:
        """
        Creates a pitch from a degree and a chromatic tone.
        """
        octave, deg = divmod ( deg, 7 )
        acci = tone - _majorScale [ deg ] - octave * 12
        return cls ( deg, acci )
    
    @classmethod
    def fromTone ( 
        cls, tone: float, 
        acciPref: AcciPref = AcciPrefs.CLOSEST_SHARP 
    ) -> Self:
        tone %= 12
        deg = acciPref ( tone )
        acci = tone - _majorScale [ deg ]
        return cls ( deg, acci )
    
    @classmethod
    def co5 ( cls, n: int = 0 ) -> Self:
        """
        Returns the `n`-th pitch in circle of fifths order, starting from C.
        positive `n` values means `n` perfect fifths up while negative `n` values means `n` 
        perfect fifths down.
        
        When `n` ranges from -7 to 7, this method yields the tonic of major scale with `abs(n)`
        sharps (for positive `n`) or flats (for negative `n`).
        """
        deg = n * 4 % 7
        acci = ( n + 1 ) // 7
        return cls ( deg, acci )
    
    def __post_init__ ( self ):
        if isinstance ( self.deg, str ):
            # input is a note name character
            object.__setattr__ ( self, "deg", ord ( self.deg.upper ( ) ) - 67 )
        if not isinstance ( self.deg, Integral ):
            raise TypeError ( f"degree must be an integer, got {self.deg.__class__.__name__}" )
        if self.deg < 0 or self.deg >= 7:
            object.__setattr__ ( self, "deg", self.deg % 7 )
    
    @property
    def tone ( self ) -> float: 
        """
        Chromatic tone of the pitch, in half-steps.
        """
        return _majorScale [ self.deg ] + self.acci
    
    @property
    def quality ( self ) -> float:
        if self.acci > 0: return self.acci + 1
        elif self.deg in _perfectIntervals:
            if self.acci == 0: return 0
            else: return self.acci - 1
        else:
            if self.acci == 0: return 1
            else: return self.acci
    
    def atOctave ( self, octave: int = 0 ) -> "Pitch":
        return Pitch ( self, octave )
    
    def isEnharmonic ( self, other: Self ) -> bool:
        return ( self.tone - other.tone ) % 12 == 0
    
    def __add__ ( self, other: Self ) -> Self:
        deg = self.deg + other.deg
        octave, deg = divmod ( deg, 7 )
        tone = self.tone + other.tone
        acci = tone - _majorScale [ deg ] - octave * 12 + self.acci
        return self.__class__ ( deg, acci )
    
    def __neg__ ( self ) -> Self:
        deg = 7 - self.deg
        tone = 12 - self.tone
        acci = tone - _majorScale [ deg ]
        return self.__class__ ( deg, acci )
    
    def __mul__ ( self, other: int ) -> Self:
        if other == 0: return self.__class__ ( )
        if not isinstance ( other, Integral ):
            raise TypeError ( f"can only multiply by integers, got {type ( other )}" )
        deg = self.deg * other
        tone = self.tone * other
        octave, deg = divmod ( deg, 7 )
        acci = tone - _majorScale [ deg ] - octave * 12 + self.acci
        return self.__class__ ( deg, acci )
    
    def __repr__ ( self ):
        return f"{self.__class__.__name__}({_noteNames [ self.deg ]}{self.acci:+.2f})"

@dataclass ( frozen = True, slots = True )
class Pitch ( PitchLike ):
    """
    Represents a pitch with specific octave, or an interval that may cross multiple octaves.
    """
    
    opitch: OPitch = OPitch ( )
    octave: int = 0 
    
    @property
    def deg ( self ) -> int:
        """
        Degree of the pitch in C major scale, not considering accidentals. Equals to the degree
        of octave pitch plus octave times 7.
        """
        return self.opitch.deg + self.octave * 7
    
    @property
    def acci ( self ) -> float:
        return self.opitch.acci
    
    @property
    def tone ( self ) -> float:
        """
        Chromatic tone of the pitch, in half-steps. Equals to the chromatic tone of octave 
        pitch plus octave times 12.
        """
        return self.opitch.tone + self.octave * 12
    
    @property
    def quality ( self ) -> float:
        return self.opitch.quality
    
    def isEnharmonic ( self, other: PitchLike ) -> bool:
        return self.tone == other.tone
    
    def hz ( self, A0: float = 440 ) -> float:
        return A0 * np.power ( 2, ( self.tone - 9 ) / 12 )
    
    def __add__ ( self, other: PitchLike ) -> Self:
        deg = self.deg + other.deg
        tone = self.tone + other.tone
        octave, odeg = divmod ( deg, 7 )
        acci = tone - _majorScale [ odeg ] - octave * 12 + self.acci
        return OPitch ( odeg, acci ).atOctave ( octave )
    
    def __neg__ ( self ) -> Self:
        return ( -self.opitch ).atOctave ( -self.octave + 1 )
    
    def __mul__ ( self, other: int ) -> Self:
        if other == 0: return self.__class__ ( )
        if not isinstance ( other, Integral ):
            raise TypeError ( f"can only multiply by integers, got {type ( other )}" )
        deg = self.deg * other
        tone = self.tone * other
        octave, odeg = divmod ( deg, 7 )
        acci = tone - _majorScale [ odeg ] - octave * 12 + self.acci
        return OPitch ( odeg, acci ).atOctave ( octave )

OInterval = OPitch
Interval = Pitch