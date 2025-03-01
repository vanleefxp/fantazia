# `fantazia`: A Math-Based Music Theory Computation Library

`fantazia` is a lightweight library for math-based music theory computation. The package is currently under development and not yet distributed to PyPI. Currently only pitch and interval computation is supported. The package name comes from "fantasia" and the preferred abbreviation is `fz`.

Different from sophisticated and feature-rich [`music21`](https://github.com/cuthbertLab/music21) library which provides a comprehensive music analysis toolkit, the target of `fantazia` is to regularize music computations by mathematic rules. The music related types in `fantazia` do not contain more information than necessary as an abstraction. Support for conversions to `music21` types is planned in the future.

# Pitch and Interval

`fantazia` provides two main pitch / interval types, which are `OPitch`, standing for a pitch with no octave specification, and `Pitch`, representing a pitch in a specific octave.

In `fantazia`, we do not distinguish between pitch and interval, because a pitch can also be regarded as an interval from middle C, just as a real number can be regarded as a point on the number line as well as a translation amount. The arithmetic expression 2 + 3 = 5, for example, can be explained by moving point 2 right by 3 steps, or moving point 3 right by 2 steps, or the combination of the operations "moving right 2 steps" and "moving right 3 steps". Here real numbers act both as a static value and an operator. For pitches and intervals, this is the same. The pitch E can stand for the static pitch and also a major 3^rd^ interval. For this reason, `fantazia` provides `OInterval` as an alias of `OPitch` and `Interval` for `Pitch`. 

```python
import fantazia as fz

# create pitch by degree and accidental
p1 = fz.OPitch ( fz.Degs.C ) # C
p2 = fz.OPitch ( "A" ) # A
p3 = fz.OPitch ( fz.Degs.B, fz.Accis.FLAT ) # B flat

print ( p1, p2, p3 )
# Output: OPitch(C+0.00) OPitch(A+0.00) OPitch(B-1.00)
```