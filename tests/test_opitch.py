import unittest
from pathlib import Path
import sys
from fractions import Fraction as Q
from importlib.metadata import PackageNotFoundError, version

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
sys.path.append(str(DIR / "../src"))

import fantazia as fz  # noqa: E402


def hasPackage(package_name):
    try:
        version(package_name)
        return True
    except PackageNotFoundError:
        return False


class TestOPitch(unittest.TestCase):
    def test_slots(self):
        types = (fz.OPitch, fz.Pitch, fz.ODeg)
        for t in types:  # the types with `__slots__` shouldn't have `__dict__`
            self.assertNotIn("__dict__", dir(t))
        with self.assertRaises(AttributeError):
            fz.OPitch("C").x = 1  # type: ignore

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

    @unittest.skipUnless(hasPackage("music21"), "requires `music21` to be installed")
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

    def test_interval(self):
        testData = map(
            lambda t: (fz.OPitch(t[0]), t[1]),
            (
                ("C", "P1"),
                ("D-", "m2"),
                ("D", "M2"),
                ("D+", "A2"),
                ("E-", "m3"),
                ("E", "M3"),
                ("F", "P4"),
                ("F[-1/2]", "m4"),
                ("F+", "A4"),
                ("F[+1/2]", "M4"),
                ("G-", "d5"),
                ("G[-1/2]", "m5"),
                ("G", "P5"),
                ("G[+1/2]", "M5"),
                ("A-", "m6"),
                ("A", "M6"),
                ("B--", "d7"),
                ("B-", "m7"),
                ("B", "M7"),
            ),
        )
        for p, ans in testData:
            self.assertEqual(p.interval(), ans)

    def test_fromTone(self):
        testData = {
            k: map(lambda t: (t[0], fz.OPitch(t[1])), v)
            for k, v in zip(
                (
                    fz.AcciPrefs.SHARP,
                    fz.AcciPrefs.FLAT,
                    fz.AcciPrefs.CLOSEST_SHARP,
                    fz.AcciPrefs.CLOSEST_FLAT,
                    fz.AcciPrefs.CLOSEST_FLAT_F_SHARP,
                ),
                (
                    (
                        (1, "C+"),
                        (3, "D+"),
                        (Q("11/2"), "F[+1/2]"),
                        (6, "F+"),
                        (Q("13/2"), "F[+3/2]"),
                        (8, "G+"),
                        (10, "A+"),
                    ),
                    (
                        (1, "D-"),
                        (3, "E-"),
                        (Q("11/2"), "G[-3/2]"),
                        (6, "G-"),
                        (Q("13/2"), "G[-1/2]"),
                        (8, "A-"),
                        (10, "B-"),
                    ),
                    (
                        (1, "C+"),
                        (3, "D+"),
                        (Q("11/2"), "F[+1/2]"),
                        (6, "F+"),
                        (Q("13/2"), "G[-1/2]"),
                        (8, "G+"),
                        (10, "A+"),
                    ),
                    (
                        (1, "D-"),
                        (3, "E-"),
                        (Q("11/2"), "F[+1/2]"),
                        (6, "G-"),
                        (Q("13/2"), "G[-1/2]"),
                        (8, "A-"),
                        (10, "B-"),
                    ),
                    (
                        (1, "D-"),
                        (3, "E-"),
                        (Q("11/2"), "F[+1/2]"),
                        (6, "F+"),
                        (Q("13/2"), "G[-1/2]"),
                        (8, "A-"),
                        (10, "B-"),
                    ),
                ),
            )
        }

        for acciPref, data in testData.items():
            with self.subTest(acciPref=acciPref):
                for tone, res in data:
                    self.assertEqual(fz.OPitch._fromTone(tone, acciPref), res)


if __name__ == "__main__":
    unittest.main()
