# `fantazia`: A Math-Based Music Theory Computation Library

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fvanleefxp%2Ffantazia%2Fmaster%2Fpyproject.toml) ![GitHub License](https://img.shields.io/github/license/vanleefxp/fantazia)

`fantazia` is a Python library for *math-based* music theory computation.

Different from sophisticated and feature-rich [`music21`](https://github.com/cuthbertLab/music21) which provides a comprehensive music analysis toolkit, the target of `fantazia` is to *regularize music computations by mathematic rules*. The music related types in `fantazia` do not contain more information than necessary as an abstraction (like color and display style, which are kept in `music21` objects), and are immutable (`music21` objects are mutable). Support for conversion to and from `music21` types is currently implemented only for pitch objects.

> The package is currently under development and not yet distributed to PyPI. You may download the `.whl` or `.tar.gz` distribution file in the `dist` sub-directory and install from it to have a try.

## Pitches / Intervals

```py
import fantazia as fz

# general pitch with no octave specification
print(
    fz.oP("C"),   # C
    fz.oP("G"),   # G
    fz.oP("F+"),  # F sharp
    fz.oP("E-"),  # E flat
    fz.oP("A--"), # A double flat
    fz.oP("D++"), # D double sharp
)
# Output: C G F+ E- A-- D++

# octave specific pitch
print(
    fz.P("C_0"),   # C in middle octave (middle C)
    fz.P("E-"),    # E flat in middle octave (octave symbol omitted)
    fz.P("F_1"),   # F at 1 octave higher from middle octave
    fz.P("G_-1"),  # G at 1 octave lower from middle octave
    fz.P("F+_2"),  # F sharp 2 octaves higher
    fz.P("A-_-1"), # A flat 1 octave lower
)
# Note: middle octave is octave 0
# Output: C_0 E-_0 F_1 G_-1 F+_2 A-_-1
```

`fantazia` provides two main pitch / interval types, which are `OPitch` (or abbreviated as `oP`), standing for a **general pitch** with no octave specification, and `Pitch` (`P`), representing a **specific pitch** in a specific octave. These two pitch types can be created from string notations.

Unlike `music21`, `fantazia` do not distinguish between pitchs and intervals by two different types. Any pitch object can also be also regarded as an interval starting from middle C.


### Pitch / Interval Properties

```py
import fantazia as fz

# general pitch
p1 = fz.oP("F+")
print(
    p1.step, # diatonic step as an integer between 0 and 6
    p1.acci, # accidental, a real number notating chromatic tone change
    p1.tone, # chromatic tone in an octave (maps an octave to [0, 12))
    p1.freq, # frequency relative to C (C as 1)
)
# Output: 3 1 6 1.4142135623730951

# specific pitch
p2 = fz.P("F+_2")
print(
    p2.opitch, # general pitch without octave specification
    p2.o,      # the octave this pitch is located in (middle octave is 0)
    p2.step,   # equals to `p2.opitch.step + p2.o * 7`
    p2.acci,   # equals to `p2.opitch.acci`
    p2.tone,   # equals to `p2.opitch.tone + p2.o * 12`
    p2.freq,   # equals to `p2.opitch.freq * 2**p2.o`
)
# Output: F+ 2 17 1 30 5.656854249492381
```

Pitch / interval objects contains a set of basic properties including step, accidental, tone and frequency, etc.

In an `OPitch` object:

* `step` is the diatonic step of the pitch object, an integer between 0 and 6, which stands for C, D, E, F, G, A and B, respectively.
* `acci` is the accidental, represented as a number (in most cases integers) notating the chromatic tone change.
* `tone` is the chromatic tone of a pitch in an octave. The size of an octave is mapped evenly to the [0, 12) range, with correspondence to the 12 semitones in an octave, where the C pitch is mapped to 0.
* `freq` is the frequency value of the pitch relative to C pitch (not the frequency in hertz). The C pitch has frequency 1.

In a `Pitch` object:

* The property `opitch` gives the general pitch of this pitch object without octave specification.
* `o` is the octave the pitch object is located in. The middle octave (of the middle C pitch) is octave 0. (This is different from the convention of MIDI, where middle octave is mostly referred to as octave 4)
* The `step` property is added by number of octaves times 7 compared to `OPitch`.
* `acci` is the same as `OPitch`
* The `tone` property is added by number of octaves times 12 compared to `OPitch`.
* `freq` is the frequency value relative to middle C. Octaves are multiplied as powers of two compared to `OPitch` objects.


### Pitch / Interval Arithmetics

```py
import fantazia as fz

# general pitch
print(
    fz.oP("E") + fz.oP("E-"),  # addition (transposing a pitch up by an interval)
    fz.oP("A") - fz.oP("D+"),  # subtraction (transposing down by an interval)
    fz.oP("E") * 2,            # multiplication by integer
    3 * fz.oP("G"),            # (order does not matter)
    -fz.oP("E"),               # negation (interval inversion)
    -fz.oP("F+"),
)
# all results will be rounded into an octave
# Output: G E- G+ A A- G-

# specific pitch
print(
    fz.P("E") + fz.P("E-"),    # addition
    fz.P("A") - fz.P("D+"),    # subtraction
    fz.P("E") * 2,             # multiplication by integer
    3 * fz.P("G"),
    -fz.P("E"),                # negation
    -fz.P("F+"),
)
# result is octave sensitive
# octave differences are kept
# Output: G_0 G-_0 G+_0 A_1 A-_-1 G-_-1
```

In `fantazia`, arithmetics between pitch objects can be done easily by operators. The most commonly used arithmetic operations include addition, negation, subtraction and multiplication by integer.

* Addition / subtraction moves a pitch up / down by an interval.
* Negation gives the inversion of an interval.
* Multiplication stacks an interval multiple times.
* In `OPitch`, the result is always taken modulus into an octave, while in `Pitch`, octave changes will be reflected in the results.


## Xenharmonic Support

While the main package of `fantazia` uses pitch notations based on the 12 tone equal temperament system, alternative tuning methods are also supported through the `fantazia.xen` subpackage. Currently available tuning methods include all equal temperaments and just intonation. Support for notating pitches with radical frequencies is being considered.

Details for alternative tunings are explained in the documentations (although currently none up till now).

```py
from fantazia.xen import edo53, ji

# equal temperaments other than 12-edo
# follows the chain-of-fifth notation rule
p1 = edo53.oP("C+")
print(
    edo53.edo,        # number of equal divisions of an octave
    edo53.sharpness,  # number of edo steps a sharp sign raises
    edo53.diatonic,   # edo steps that the diatonic steps are mapped to
)
# Output: 53 5 [ 0  9 18 22 31 40 49]

print(
    p1, p1.step, p1.acci,
    p1.tone,  # `tone` is measured by edo steps here
    p1.pos,   # relative position of the pitch in an octave
              # where an octave is mapped evenly to [0, 1)
    p1.freq,
)
# Output: C+@edo53 0 1 5 5/53 1.067576625048014

# just intonation, where frequencies are rational numbers
# follows the FJS (functional just system) notation rule
p2 = ji.oP("E(5)")
print(
    p2, p2.step, p2.acci, p2.pos,
    p2.freq,  # frequency represented in exact form as a rational number
)
# Output: E(5)@ji3 2 0 0.32192809488736235 5/4
```