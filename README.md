# `fantazia`: A Math-Based Music Theory Computation Library

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fvanleefxp%2Ffantazia%2Fmaster%2Fpyproject.toml) ![GitHub License](https://img.shields.io/github/license/vanleefxp/fantazia)

`fantazia` is a Python library for *math-based* music theory computation.

Different from sophisticated and feature-rich [`music21`](https://github.com/cuthbertLab/music21) which provides a comprehensive music analysis toolkit, the target of `fantazia` is to *regularize music computations by mathematic rules*. The music related types in `fantazia` do not contain more information than necessary as an abstraction (like color and display style), and are immutable. Support for conversion to `music21` types is currently implemented only for pitch objects.

> The package is currently under development and not yet distributed to PyPI. You may download the `.whl` or `.tar.gz` distribution file in the `dist` sub-directory and install from it to have a try.


## Getting Started

[TODO]

## Pitches / Intervals

`fantazia` provides two main pitch / interval types, which are `OPitch`, standing for a **general pitch** with no octave specification, and `Pitch`, representing a **specific pitch** in a specific octave.

Unlike `music21`, `fantazia` do not create separated types for pitches and intervals, because a pitch can also be regarded as an interval from the origin, middle C. This makes the arithmetics of pitch objects more convenient and consistent.

> This consideration has an analogy with points and vectors in Euclidean space, where a point can also be regarded as a vector starting from origin. Choosing not to distinguish between vectors and points by two different classes in programming is thus more convenient for the implementation of vector arithmetics.


### Creating Pitch / Interval Objects


#### `OPitch`: General Pitch Type

The simplest method to create an `OPitch` object is to call `OPitch()` constructor, or its equivalent abbreviation, `oP()`, with a notation string, such as `C`, `F+`, `E-`, etc.

```py
import fantazia as fz
print(fz.oP("C"), fz.oP("F+"), fz.oP("E-"))
# C F+ E-
```

A notation string consists of a step name and, a symbolic representation of accidental, if any.

A valid step name includes:

* a letter among `C`, `D`, `E`, `F`, `G`, `A`, `B`
* a solfège name among `do` / `ut`, `re`, `mi`, `fa`, `sol`, `la`, `si` / `ti`
* a degree value among `1`, `2`, `3`, `4`, `5`, `6`, `7` (note that in string representation, counting starts from one instead of zero as in the tradition of music theory)

A symbolic accidental can be:

* a combination of `+` and `-` signs, where `+` stands for sharp (&sharp;) and `-` for flat (&flat;). In this sense, `++` represents double sharp (&#x1d12a;) and `--` is for double flat (&#x1d12b;).
* A numeric value, in the form of an integer, decimal or fraction, preceded by a `+` or `-`, and wrapped in square brackets. (to support microtonal notation and simplify multiple accidentals. *e.g.* `[+1/2]`, `[-0.5]`, `[+4]`)

```python
import fantazia as fz

pitches = map(
    fz.OPitch,
    ("C", "F+", "e-", "G++", "A--", "re+", "7-")
)
print(" ".join(map(str, pitches)))
# C F+ E- G++ A-- D+ B-
```

> This notation rule is inspired by the syntax of [Alda](https://alda.io/) music programming language. While sometimes in plain text people use `#` and `b` to notate sharps and flats, `fantazia` do not opt for this notation because `b` might be confused with the note name B.


Below are the most important properties an `OPitch` object has:

* `step`: an integer between `0` and `6`
* `acci`: an arbitrary real number representing alteration from the base chromatic tone defined by `step` in semitones, which is `(0, 2, 4, 5, 7, 9, 11)` respectively for `step` value `0` to `6`. Typically integers, or, in the microtonal case, fractions
* `tone`: sum of base chromatic tone and accidental value, which refers to the tonal distance of this `OPitch` object from the C pitch in semitones. The identity holds that `p.tone == (0, 2, 4, 5, 7, 9, 11)[p.step] + p.acci` (or, equivalently)


<!-- The `OPitch` object has another property called `tone`, which describes the pitch's chromatic position in an octave. The octave space from C to a higher C is mapped to [0, 12) range, and notes C, D, E, F, G, A, B without accidental have tone values of 0, 2, 4, 5, 7, 9, 11, respectively.

The tone value is computed by summing the tone value of base note and the numeric representation of accidental. (i.e. `p.tone = (0, 2, 4, 5, 7, 9, 11)[p.deg] + p.acci`)

```python
# ... following previous examples
print(p1.tone, p2.tone, p3.tone)
# Output: 0 9 10
```

> Note that for consistency reasons the tone value might sometimes exceed the [0, 12) range. For example, B sharp has tone value 12 while C flat has tone value -1. However it can be ascertained that in `OPitch`, [enharmonics](#enharmonics) (i.e. notes representing the same pitch but spelled differently) have the same tone value modulus by 12.

`OPitch` objects can also be created by specifying a degree and tone value, or just a tone value. When using just a tone value, accidental will be chosen automatically according to an accidental preference rule.

```python
import fantazia as fz

p6 = fz.OPitch.fromDegAndTone("E", 3) # E flat
p7 = fz.fromDeg(6) # F sharp

print(p6, p7) # Output: E- F+
```

> The accidental preference rule is a more complicated topic and will not be described in details here. It can be set by the `acciPref` argument in `fz.fromDeg` and have some possible values listed in `fz.AcciPrefs.XXX`. -->


#### `Pitch`: Specific Pitch Type

Similar to `OPitch` objects, you can also use a string notation to create a `Pitch` object by calling the `Pitch()` constructor, or its abbreviation `P()`.

The string notation of a `Pitch` object is a valid `OPitch` string notation followed by an underscore and an octave value, such as `C_0`, `E-_2`, `F+_-1`. When the octave is omitted it is defaulted to 0.

Unlike the convention of MIDI, `fantazia` refers to the octave of middle C as octave 0 instead of octave 4.

```py
import fantazia as fz

print(fz.P("C_0"), fz.P("E-_2"), fz.P("F+_-1"), fz.P("A"))
# C_0 E-_2 F+_1 A_0
```

The most important properties of a `Pitch` object are:

* `opitch`, which refers to the general pitch this pitch belongs to without specific octave
* `o`: the nominal octave of the pitch
* `step`: the octave-sensitive step value. Equals to `p.opitch.step + 7 * p.o`
* `acci`: the accidental of the pitch, which is the same as `p.opitch.acci`
* `tone`: octave-sensitive tone value. Equals to `p.opitch.tone + 12 * p.o`


### Pitch / Interval Arithmetics


#### Addition

Adding two pitches / intervals `p1`, `p2` results in a third pitch / interval `p3` defined by the following conditions:

* In the `OPitch` case:
  * `(p1.step + p2.step) % 7 == p3.step`
  * `p1.tone + p2.tone - (p1.step + p2.step) // 7 * 12 == p3.tone`
  
* In the `Pitch` case:
  * `p1.step + p2.step == p3.step`
  * `p1.tone + p2.tone == p3.tone`

It can be argued that the pitch `p3` exists and is unique. Addition of pitches / intervals can be comprehended as stacking two intervals together or transposing a pitch by an interval. For `OPitch`, the result is taken modulus into the range of an octave, while the result of `Pitch` is octave-sensitive.

```py
import fantazia as fz

print(fz.oP("E") + fz.oP("E-"))  # G
print(fz.oP("E") + fz.oP("E"))  # G+
print(fz.oP("E-") + fz.oP("E-"))  # G-

print(fz.oP("F+") + fz.oP("A-"))  # D
print(fz.P("F+_0") + fz.P("A-_0"))  # D_1
```

#### Negation and Subtraction

#### Multiplication by Integer

#### Group Theory Explanation

Pitches / intervals exhibit [**Abelian group**](https://en.wikipedia.org/wiki/Abelian_group) properties. Below is an explanation on `OPitch` objects. For `Pitch` objects you need to take octave into consideration, but the rudiments are the same.

* two `OPitch` objects are equal *if and only if* they have the same `deg` and `acci` value, or, equivalently, the same `deg` and `tone` value.

* In the set of all possible `OPitch` objects, the **operation "add"** can be defined such that `p1 + p2` results in a new `OPitch` object `p3`, where `p3.deg == (p1.deg + p2.deg) % 7` and `p3.tone % 12 == (p1.tone + p2.tone) % 12`. Such a `p3` exists and is unique for any `p1` and `p2`. The "add" operation is commutative.

* The C pitch / perfect unison interval is the **identity element**, whose `deg` and `tone` value both equals to `0`. Adding any pitch by C result in the same return as itself, and C is the only element satisfying this property.

* When regarding `OPitch` objects as an interval, the inversion of an interval is the **negation** of the original object, because an interval and its negation sums up to perfect octave, which, in `OPitch` will be taken modulus to perfect unison.


### Enharmonics

**Enharmonics** refer to pitches with different musical notation but share the same tone value, for example, C sharp and D flat.

In `fantazia` you can judge whether two `OPitch` objects are enharmonic by using `p1.isEnharmonic(p2)`. This is equivalent to `(p1.tone - p2.tone) % 12 == 0`.

```python
import fantazia as fz

p1 = fz.OPitch("F+")
p2 = fz.OPitch("G-")
p3 = fz.OPitch("B+")
p4 = fz.OPitch("C")

print(*(p.tone for p in (p1, p2, p3, p4)))
print(p1.isEnharmonic(p2), p3.isEnharmonic(p4))

# Output:
# 6 6 12 0
# True True
```

For octave specific `Pitch` objects, `p1.isEnharmonic(p2)` is equivalent to `p1.tone == p2.tone`. The same note in different octaves are not considered enharmonics.

```python
import fantazia as fz

p1 = fz.Pitch("C_0")
p2 = fz.Pitch("B+_-1")
p3 = fz.Pitch("B+_0")

print(p1.isEnharmonic(p2), p1.isEnharmonic(p3))
# Output: True False
```

#### Group Theory Explanation

Enharmonic is an **equivalence relation** (i.e. it is reflexive, symmetric and transitive). All notes that are enharmonic of C, like C itself, B sharp, D double-flat, etc., forms a **subgroup** of the pitch Abelian group: since they all have tone value 0 (or 12), they add up to have tone value still 0 (or 12). The negation of these pitches are also enharmonic of C. By taking the **quotient group** of the pitch group over this subgroup, the result is **isomorphic** to &#x2124;<sub>12</sub> = &#x2124;/12&#x2124; (or, if taking microtones into consideration, &#x211d;/12&#x2124;).