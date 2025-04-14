import unittest
from pathlib import Path
import sys
from fractions import Fraction as Q

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
sys.path.append(str(DIR / "../src"))

import fantazia as fz  # noqa: E402


@lambda _: _()
def HAS_M21():
    try:
        # fmt: off
        import music21  # noqa: F401
        return True
        # fmt: on
    except ImportError:
        return False


class TestOPitch(unittest.TestCase):
    def test_parse(self):
        # create `OPitch` from string representation
        testData = (
            ("C", fz.OPitch.C),
            ("F+", fz.OPitch(3, 1)),
            ("G++", fz.OPitch(4, 2)),
            ("D+++", fz.OPitch(1, 3)),
            ("E-", fz.OPitch(2, -1)),
            ("A--", fz.OPitch(5, -2)),
            ("G---", fz.OPitch(4, -3)),
            # microtonal notation
            ("F[+1/2]", fz.OPitch(3, Q(1, 2))),
            ("B[-1/2]", fz.OPitch(6, Q(-1, 2))),
        )
        for src, ans in testData:
            self.assertEqual(fz.OPitch(src), ans)
            self.assertEqual(fz.OPitch(src.lower()), ans)  # case insensitive

        with self.assertRaises(ValueError):
            fz.OPitch("XXX")  # invalid pitch name

    def test_co5(self):
        # generate pitches by circle of fifth order
        co5SharpPitches = map(
            fz.OPitch,
            ("C", "G", "D", "A", "E", "B", "F+", "C+"),
        )
        co5FlatPitches = map(
            fz.OPitch,
            ("C", "F", "B-", "E-", "A-", "D-", "G-", "C-"),
        )
        fifth = fz.OPitch("G")
        for i, ans in enumerate(co5SharpPitches):
            self.assertEqual(fz.OPitch.co5(i), ans)
            self.assertEqual(fifth * i, ans)
        for i, ans in enumerate(co5FlatPitches):
            self.assertEqual(fz.OPitch.co5(-i), ans)
            self.assertEqual(fifth * (-i), ans)

    def test_add(self):
        testData = map(
            lambda t: tuple(map(fz.OPitch, t)),
            (
                ("E", "E-", "G"),
                ("E", "E", "G+"),
                ("E-", "E-", "G-"),
            ),
        )
        for p1, p2, p3 in testData:
            self.assertEqual(p1 + p2, p3)
            self.assertEqual(p2 + p1, p3)  # commutative property

    def test_neg(self):
        testData = map(
            lambda t: tuple(map(fz.OPitch, t)),
            (
                ("C", "C"),
                ("D", "B-"),
                ("E", "A-"),
                ("F", "G"),
                ("F+", "G-"),
                ("G", "F"),
                ("A", "E-"),
                ("B", "D-"),
            ),
        )
        for p1, p2 in testData:
            self.assertEqual(-p1, p2)
            self.assertEqual(-p2, p1)
            self.assertEqual(-(-p1), p1)
            self.assertEqual(-(-p2), p2)
            self.assertEqual(p1 + p2, fz.OPitch.C)

    def test_enharmonic(self):
        testData = map(
            lambda t: tuple(map(fz.OPitch, t)),
            (
                ("C+", "D-"),
                ("F+", "G-"),
                ("F", "E+"),
                ("E", "F-"),
                ("B", "C-"),
                ("C", "B+"),
                ("G", "F++"),
                ("G", "A--"),
            ),
        )
        for p1, p2 in testData:
            self.assertTrue(p1.isEnharmonic(p2))
            self.assertTrue(p2.isEnharmonic(p1))

    def test_respell(self):
        testData = map(
            lambda t: (fz.OPitch(t[0]), t[1], fz.OPitch(t[2])),
            (
                ("C", 1, "D--"),
                ("C", -1, "B+"),
                ("C+", 1, "D-"),
                ("D", 1, "E--"),
                ("D", -1, "C++"),
                ("D+", 1, "E-"),
                ("E", 1, "F-"),
                ("E", -1, "D++"),
                ("F", 1, "G--"),
                ("F", -1, "E+"),
            ),
        )
        for p1, degAlt, p2 in testData:
            self.assertEqual(p1.respell(degAlt), p2)
            self.assertEqual(p2.respell(-degAlt), p1)

    @unittest.skipUnless(HAS_M21, "requires `music21` to be installed")
    def test_m21(self):
        from music21.pitch import Pitch as m21Pitch

        testData = map(
            lambda src: (fz.OPitch(src), m21Pitch(src.replace("+", "#"))),
            (
                "C",
                "C+",
                "D-",
                "D",
                "D+",
                "E-",
                "E",
                "F",
                "F+",
                "G-",
                "G",
                "G+",
                "A-",
                "A",
                "A+",
                "B-",
                "B",
            ),
        )
        for p, m21p in testData:
            self.assertEqual(p.m21(), m21p)


if __name__ == "__main__":
    unittest.main()
