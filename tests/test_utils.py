import unittest
from pathlib import Path
import sys

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
sys.path.append(str(DIR / "../src"))

from fantazia.utils import rdivmod  # noqa: E402


class TestUtils(unittest.TestCase):
    def test_rdivmod(self):
        testData = (
            (5, 12),
            (9, 12),
            (12, 12),
            (-5, 12),
            (-9, 12),
            (6, 12),
            (-6, 12),
        )
        ans = {
            "f": (
                (0, 5),
                (0, 9),
                (1, 0),
                (-1, 7),
                (-1, 3),
                (0, 6),
                (-1, 6),
            ),
            "c": (
                (1, -7),
                (1, -3),
                (1, 0),
                (0, -5),
                (0, -9),
                (1, -6),
                (0, -6),
            ),
            "u": (
                (1, -7),
                (1, -3),
                (1, 0),
                (-1, 7),
                (-1, 3),
                (1, -6),
                (-1, 6),
            ),
            "d": (
                (0, 5),
                (0, 9),
                (1, 0),
                (0, -5),
                (0, -9),
                (0, 6),
                (0, -6),
            ),
        }
        for rmode, expected in ans.items():
            for (n, d), res in zip(testData, expected):
                with self.subTest(n=n, d=d, rmode=rmode):
                    self.assertEqual(rdivmod(n, d, rmode=rmode, round=False), res)


if __name__ == "__main__":
    unittest.main()
