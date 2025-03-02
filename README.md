# `fantazia`: A Math-Based Music Theory Computation Library

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fvanleefxp%2Ffantazia%2Fmaster%2Fpyproject.toml) ![GitHub License](https://img.shields.io/github/license/vanleefxp/fantazia)
 [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

`fantazia` is a lightweight library for math-based music theory computation. The package is currently under development and not yet distributed to PyPI. Currently only pitch and interval computation is supported. The package name comes from "fantasia" and the preferred abbreviation is `fz`.

Different from sophisticated and feature-rich [`music21`](https://github.com/cuthbertLab/music21) library which provides a comprehensive music analysis toolkit, the target of `fantazia` is to regularize music computations by mathematic rules. The music related types in `fantazia` do not contain more information than necessary as a mathematic abstraction, and are immutable. Support for conversion to `music21` types is planned in the future.

## Pitch and Interval

`fantazia` provides two main pitch / interval types, which are `OPitch`, standing for a pitch with no octave specification, and `Pitch`, representing a pitch in a specific octave.

In `fantazia`, we do not distinguish between pitch and interval, because a pitch can also be regarded as an interval from middle C, just as a real number can be regarded as a point on the number line as well as a translation amount. The arithmetic expression 2 + 3 = 5, for example, can be explained by moving point 2 right by 3 steps, or moving point 3 right by 2 steps, or the combination of the operations "moving right 2 steps" and "moving right 3 steps". Here real numbers act both as a static value and an operator. For pitches and intervals, this is the same. The pitch E can stand for the static pitch and also a major third interval. For this reason, `fantazia` provides `OInterval` as an alias of `OPitch` and `Interval` for `Pitch`. 

### Creating Pitch / Interval Objects

The basic method to create an `OPitch` object is to call `OPitch()` constructor with a specified degree and accidental. The degree is an integer between 0 and 6 (both inclusive), standing for the note names C, D, E, F, G, A, B, respectively. The note name character is also accepted. Degree and accidental can be accessed by the `deg` and `acci` properties of `OPitch`. Accidental is a numeric value notating the pitch alteration from the standard pitch in semitones. When accidental is omitted, it defaults to natural (no accidental).

```python
import fantazia as fz

# create pitch by degree and accidental
p1 = fz.OPitch(fz.Degs.C) # C
p2 = fz.OPitch("A") # A
p3 = fz.OPitch(fz.Degs.B, fz.Accis.FLAT) # B flat

print(p1, p2, p3, p3.deg, p3.acci) # Output: C A B- 6 -1
```

The `OPitch` object has another property called `tone`, which describes the pitch's chromatic position in an octave. The octave space is mapped to [0, 12) range, and notes C, D, E, F, G, A, B without accidental have tone values of 0, 2, 4, 5, 7, 9, 11, respectively. The tone value is computed by summing the tone value of base note and the numeric representation of accidental. (i.e. `p.tone = (0, 2, 4, 5, 7, 9, 11)[p.deg] + p.acci`)

```python
print(p1.tone, p2.tone, p3.tone)
# Output: 0 9 10
```

`OPitch` objects can also be created by specifying a degree and tone value, or just a tone value. When using just a tone value, an accidental will be chosen automatically according to the `acciPref` argument.

```python
import fantazia as fz

p4 = fz.OPitch.fromDegAndTone("E", 3) # E flat
p5 = fz.fromDeg(6) # F sharp

print(p4, p5) # Output: E- F+
```

### Pitch / Interval Calculations

Pitches / intervals exhibit [**Abelian group**](https://en.wikipedia.org/wiki/Abelian_group) properties. Below is an explanation on `OPitch` objects. For `Pitch` objects you need to take octave into consideration, but the rudiments are the same.

* two `OPitch` objects are equal *if and only if* they have the same `deg` and `acci` value, or, equivalently, the same `deg` and `tone` value.

* In the set of all possible `OPitch` objects, the **operation "add"** can be defined such that `p1 + p2` results in a new `OPitch` object `p3`, where `p3.deg == (p1.deg + p2.deg) % 7` and `p3.tone % 12 == (p1.tone + p2.tone) % 12`. Such a `p3` exists and is unique for any `p1` and `p2`. The "add" operation is commutative.

* The C pitch / perfect unison interval is the **identity element**, whose `deg` and `tone` value both equals to `0`. Adding any pitch by C result in the same return as itself, and C is the only element satisfying this property.

* When regarding `OPitch` objects as an interval, the inversion of an interval is the **negation** of the original object, because an interval and its negation sums up to perfect octave, which, in `OPitch` will be taken modulus to perfect unison.

`fantazia` fully supports the addition, subtraction, negation and integer multiplication according to the above-mentioned rules.

```python
import fantazia as fz

p1 = fz.OPitch(2) # E or M3
p2 = fz.OPitch(2, -1) # E flat or m3
print(p1 + p2, p1 - p2, p1 * 2, -p1)

# Output: G C+ G+ A-

# Explanation:
# M3 + m3 = P5, M3 - m3 = A1, M3 * 2 = A5, -M3 = m6
```
