import unittest
import sympy as sym
from _routh_array import RouthArray


class test_RouthArray(unittest.TestCase):

    def test_format_poly(self):
        # define symbol
        x = sym.symbols('x')
        p = (x + 5)*(x + 3)
        res = RouthArray.format_poly(RouthArray, p, x)
        exp = (x**2) + (8*x) + 15
        self.assertEqual(res, exp)

    def test_get_coeffs(self):
        # define symbol
        x = sym.symbols('x')
        p = (x + 5)*(x + 3)
        p = RouthArray.format_poly(RouthArray, p, x)
        res = RouthArray.get_coeffs(RouthArray, p, x)
        exp = [1, 8, 15]
        self.assertEqual(res, exp)

        # verify it works with non polynomials
        res = RouthArray.get_coeffs(RouthArray, 5, x)
        exp = [5]
        self.assertEqual(res, exp)

    def test_compute_array(self):
        x = sym.symbols('x')
        p = (x + 5)*(x + 3)
        ra = RouthArray(p, x)
        res = ra.array
        exp = sym.Matrix([[1, 15], [8, 0], [15, 0]])
        self.assertEqual(res, exp)

    def test_print_array(self):
        x, K, y = sym.symbols('x, K, y')
        p = (x + K)*(x + 10)*(x + 5)*(x + y)
        arr = RouthArray(p, x)
        arr.set_print_width(99)  # num divisable by 3
        text = str(arr)

        # ensure all lines are the same length
        unique_lengths = 1
        self.assertEqual(
            len(set(map(len, text.splitlines()))),
            unique_lengths
        )

        # ensure that that length is equal to the
        # print width attribute
        line_length = arr.print_width
        self.assertEqual(
            len(text.splitlines()[0]), line_length
        )

    def test_compute_bad_array(self):
        """Test to verify arrays that will
        cause zero division errors are handled"""
        x = sym.symbols('x')
        p = (x**4) + (x**3) + (x**2) + x + 1
        array = RouthArray(p, x)
        self.assertEqual(array(-1, 0), sym.nan)

    def test_get_domain(self):
        x, K, y = sym.symbols('x, K, y')

        # successful if it computes with out error
        p = (x + 2)*(x + K)
        RouthArray(p, x).domain

        # successful it it throws error
        p = (x + 2)*(x + K)*(x - y)
        with self.assertRaises(NotImplementedError):
            RouthArray(p, x).domain


if __name__ == "__main__":
    unittest.main(verbosity=True)
