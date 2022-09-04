import sys
sys.path.append("..")


import unittest
import numpy as np
from numeric_ad.binary_operators import *


class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        actual = y.data
        expected = np.array(5.0)
        self.assertEqual(actual, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        y.backward()

        actual_0 = x0.grad
        expected_0 = np.array(1.0)
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = np.array(1.0)
        self.assertEqual(actual_1, expected_1)