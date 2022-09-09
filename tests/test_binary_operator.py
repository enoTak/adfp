import sys
sys.path.append("../.")


import unittest
import numpy as np
from numeric_ad.binary_operators import *
from numeric_ad import Variable


class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 + x1
        actual = y.data
        expected = np.array(5.0)
        self.assertEqual(actual, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 + x1
        y.backward()

        actual_0 = x0.grad
        expected_0 = np.array(1.0)
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = np.array(1.0)
        self.assertEqual(actual_1, expected_1)

    def test_combination_with_np_array(self):
        x0 = Variable(np.array(2.0))
        x1 = np.array(3.0)
        y = x0 + x1
        actual = y.data
        expected = np.array(5.0)
        self.assertEqual(actual, expected)

    def test_combination_with_np_array_with_list(self):
        x0 = np.array([3.0])
        x1 = Variable(np.array([2.0]))
        y = x0 + x1
        actual = y[0].data
        expected = np.array([5.0])
        self.assertEqual(actual, expected)

    def test_combination_with_float(self):
        x0 = Variable(np.array(2.0))
        x1 = 3.0
        y = x0 + x1
        actual = y.data
        expected = np.array(5.0)
        self.assertEqual(actual, expected)

        x0 = 3.0
        x1 = Variable(np.array(2.0))
        y = x0 + x1
        actual = y.data
        expected = np.array(5.0)
        self.assertEqual(actual, expected)


class MulTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 * x1
        actual = y.data
        expected = np.array(6.0)
        self.assertEqual(actual, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 * x1
        y.backward()

        actual_0 = x0.grad
        expected_0 = np.array(3.0)
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = np.array(2.0)
        self.assertEqual(actual_1, expected_1)

    def test_combination_with_np_array(self):
        x0 = Variable(np.array(2.0))
        x1 = np.array(3.0)
        y = x0 * x1
        actual = y.data
        expected = np.array(6.0)
        self.assertEqual(actual, expected)

    def test_combination_with_float(self):
        x0 = Variable(np.array(2.0))
        x1 = 3.0
        y = x0 * x1
        actual = y.data
        expected = np.array(6.0)
        self.assertEqual(actual, expected)

        x0 = 3.0
        x1 = Variable(np.array(2.0))
        y = x0 * x1
        actual = y.data
        expected = np.array(6.0)
        self.assertEqual(actual, expected)