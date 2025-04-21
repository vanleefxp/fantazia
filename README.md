# `fantazia`: A Math-Based Music Theory Computation Library

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fvanleefxp%2Ffantazia%2Fmaster%2Fpyproject.toml) ![GitHub License](https://img.shields.io/github/license/vanleefxp/fantazia)

`fantazia` is a Python library for *math-based* music theory computation. 

Different from sophisticated and feature-rich [`music21`](https://github.com/cuthbertLab/music21) which provides a comprehensive music analysis toolkit, the target of `fantazia` is to *regularize music computations by mathematic rules*. The music related types in `fantazia` do not contain more information than necessary as an abstraction (like color and display style), and are immutable. Support for conversion to `music21` types is currently implemented only for pitch objects.

> The package is currently under development and not yet distributed to PyPI. You may download the `.whl` or `.tar.gz` distribution file in the `dist` sub-directory and install from it to have a try.


## Getting Started

[TODO]

## Pitch and Interval

`fantazia` provides two main pitch / interval types, which are `OPitch`, standing for a **pitch class** (pitch with no octave specification), and `Pitch`, representing a **pitch** in a specific octave.

Unlike `music21`, `fantazia` do not create different types for pitches and intervals, because a pitch can also be regarded as an interval from middle C. This makes the arithmetic of pitch objects more convenient and consistent.

> This consideration has an analogy with points and vectors in Euclidean space, where a point can also be regarded as a vector starting from origin. Distinguishing between vectors and points is hence not necessary.

<!-- , just as a real number can be regarded as a point on the number line as well as a translation amount. The arithmetic expression 2 + 3 = 5, for example, can be explained by moving point 2 right by 3 steps, or moving point 3 right by 2 steps, or the combination of the operations "moving right 2 steps" and "moving right 3 steps". Here real numbers act both as a static value and an operator. For pitches and intervals, this is the same. The pitch E can stand for the static pitch and also a major third interval. For this reason, `fantazia` provides `OInterval` as an alias of `OPitch` and `Interval` for `Pitch`.  -->


### Creating Pitch / Interval Objects


#### `OPitch`: pitch without specific octave

The simplest method to create an `OPitch` object is to call `OPitch()` constructor with a notation string. A notation string consists of a step name and, a symbolic representation of accidental, if any.

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

> This notation rule is inspired by the syntax of [Alda](https://alda.io/) music programming language. While sometimes in plain text people use `#` and `b` to notate sharps and flats, I do not opt for this notation because `b` might be confused with the note name B.

The `OPitch` object has another property called `tone`, which describes the pitch's chromatic position in an octave. The octave space from C to a higher C is mapped to [0, 12) range, and notes C, D, E, F, G, A, B without accidental have tone values of 0, 2, 4, 5, 7, 9, 11, respectively. 

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

> The accidental preference rule is a more complicated topic and will not be described in details here. It can be set by the `acciPref` argument in `fz.fromDeg` and have some possible values listed in `fz.AcciPrefs.XXX`.


#### `Pitch`: pitch with specific octave

[TODO]


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


### Enharmonics

**Enharmonics** refer to pitches with different musical notation but share the same tone value, for example, C sharp and D flat.

Enharmonic is an **equivalence relation**, i.e. it is reflexive, symmetric and transitive. All notes that are enharmonic of C, like C itself, B sharp, D double-flat, etc., forms a **subgroup** of the pitch Abelian group: since they all have tone value 0 (or 12), they add up to have tone value still 0 (or 12). The negation of these pitches are also enharmonic of C. By taking the **quotient group** of the pitch group over this subgroup, the result is **isomorphic** to &#x2124;<sub>12</sub> = &#x2124;/12&#x2124; (or, if taking microtones into consideration, &#x211d;/12&#x2124;).

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