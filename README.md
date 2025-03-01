# `fantazia`: A Math-Based Music Theory Computation Library

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`fantazia` is a lightweight library for math-based music theory computation. The package is currently under development and not yet distributed to PyPI. Currently only pitch and interval computation is supported. The package name comes from "fantasia" and the preferred abbreviation is `fz`.

Different from sophisticated and feature-rich [`music21`](https://github.com/cuthbertLab/music21) library which provides a comprehensive music analysis toolkit, the target of `fantazia` is to regularize music computations by mathematic rules. The music related types in `fantazia` do not contain more information than necessary as an abstraction. Support for conversions to `music21` types is planned in the future.

## Pitch and Interval

`fantazia` provides two main pitch / interval types, which are `OPitch`, standing for a pitch with no octave specification, and `Pitch`, representing a pitch in a specific octave.

In `fantazia`, we do not distinguish between pitch and interval, because a pitch can also be regarded as an interval from middle C, just as a real number can be regarded as a point on the number line as well as a translation amount. The arithmetic expression 2 + 3 = 5, for example, can be explained by moving point 2 right by 3 steps, or moving point 3 right by 2 steps, or the combination of the operations "moving right 2 steps" and "moving right 3 steps". Here real numbers act both as a static value and an operator. For pitches and intervals, this is the same. The pitch E can stand for the static pitch and also a major 3^rd^ interval. For this reason, `fantazia` provides `OInterval` as an alias of `OPitch` and `Interval` for `Pitch`. 

### Creating Pitch Objects

The basic method to create an `OPitch` object is to call `OPitch()` constructor with a specified degree and accidental. The degree is an integer between 0 and 6 (both inclusive), standing for the note names C, D, E, F, G, A, B, respectively. The note name character is also accepted. Degree and accidental can be accessed by the `deg` and `acci` properties of `OPitch`. Accidental is a numeric value notating the pitch alteration from the standard pitch in semitones. When accidental is omitted, it defaults to natural (no accidental).

```python
import fantazia as fz

# create pitch by degree and accidental
p1 = fz.OPitch(fz.Degs.C) # C
p2 = fz.OPitch("A") # A
p3 = fz.OPitch(fz.Degs.B, fz.Accis.FLAT) # B flat

print(p1, p2, p3)
print(p3.deg, p3.acci)
# Output: 
# OPitch(C+0.00) OPitch(A+0.00) OPitch(B-1.00)
# 6 -1
```

The `OPitch` object has another property called `tone`, which describes the pitch's chromatic position in an octave. The octave space is mapped to [0, 12) range, and notes C, D, E, F, G, A, B without accidental have tone values of 0, 2, 4, 5, 7, 9, 11, respectively. The tone value is computed by summing the tone value of base note and the numeric representation of accidental. (i.e. `p.tone = (0, 2, 4, 5, 7, 9, 11)[p.deg] + p.acci`)

```python
print(p1.tone, p2.tone, p3.tone)
# Output: 0 9 10
```

