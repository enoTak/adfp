import unittest
import numpy as np
from adfp import Variable
from adfp.calc_utils import sum_to

class SumToTest(unittest.TestCase):
    def test_value(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = sum_to(x, (1, 3))
        actual = y
        expected = np.array([[5, 7, 9]])
        self.assertTrue(np.array_equal(actual, expected))

        y = sum_to(x, (2, 1))
        actual = y
        expected = np.array([[6], [15]])
        self.assertTrue(np.array_equal(actual, expected))