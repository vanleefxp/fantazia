"""
# `fantazia`: a Math-Based Music Theory Computation Library

This is the top-level module of the `fantazia` library. You can access the basic music theory
types from here, such as pitch / interval and mode.

```ebnf
degree-notation   = scale-degree, acci;
opitch-notation   = note-name, acci;
pitch-notation    = opitch-notation, ["_", [sign], integer];
note-name         = letter-name | solfege-name | scale-degree;
letter-name       = "C" | "D" | "E" | "F" | "G" | "A" | "B";
solfege-name      = "do" | "re" | "mi" | "fa" | "sol" | "la" | "si" | "ti" ;
(* `letter-name` and `solfege-name` are case insensitive *)
scale-degree      = "1" | "2" | "3" | "4" | "5" | "6" | "7";
acci              = "" | symbolic-acci | numeric-acci
non-zero-digit    = "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9";
digit             = "0" | non-zero-digit;
positive-integer  = non-zero-digit, {digit};
natural-number    = "0" | positive-integer;
positive-decimal  = [integer], ".", {digit};
fraction          = natural-number | "/" | positive-integer;
sign              = "+" | "-";
symbolic-acci     = {sign};
numeric-acci      = "[", sign, natural-number | fracion | positive-decimal, "]";
```

```ebnf
interval-notation = [sign], quality, natural-number
quality           = quality-base, [numeric-acci]
quality-base      = ("A", {"A"}) | "M" | "P" | "m" | ("d", {"d"}) | multiple-aug-dim
(* `quality-base` is case sensitive *)
multiple-aug-dim  = ("[", "A" | "d", "*", natural-number, "]")
```
"""

from ._theory import *  # noqa: F401, F403
from . import xen  # noqa: F401
