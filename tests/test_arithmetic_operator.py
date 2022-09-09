import sys
sys.path.append("../.")


import unittest
import numpy as np
from autodiff.arithmetic_operator import *
from autodiff import Variable


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
        actual = y.data
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

    def test_combination_with_np_array_with_list(self):
        x0 = np.array([3.0])
        x1 = Variable(np.array([2.0]))
        y = x0 * x1
        actual = y.data
        expected = np.array([6.0])
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


class NegTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = -x
        actual = y.data
        expected = np.array(-2.0)
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = -x
        y.backward()

        actual = x.grad
        expected = np.array(-1.0)
        self.assertEqual(actual, expected)


class SubTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(1.0))
        y = x0 - x1
        actual = y.data
        expected = np.array(1.0)
        self.assertEqual(actual, expected)

        x = Variable(np.array(2.0))
        y = x - 4.0
        actual = y.data
        expected = np.array(-2.0)
        self.assertEqual(actual, expected)

        y = 5.0 - x
        actual = y.data
        expected = np.array(3.0)
        self.assertEqual(actual, expected)


    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 - x1
        y.backward()

        actual_0 = x0.grad
        expected_0 = np.array(1.0)
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = np.array(-1.0)
        self.assertEqual(actual_1, expected_1)


class DivTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(2.0))
        y = x0 / x1
        actual = y.data
        expected = np.array(1.0)
        self.assertEqual(actual, expected)
        
        x = Variable(np.array(2.0))
        y = x / 4.0
        actual = y.data
        expected = np.array(0.5)
        self.assertEqual(actual, expected)

        y = 5.0 / x
        actual = y.data
        expected = np.array(2.5)
        self.assertEqual(actual, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 - x1
        y.backward()

        actual_0 = x0.grad
        expected_0 = np.array(1.0)
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = np.array(-1.0)
        self.assertEqual(actual_1, expected_1)

    def test_zero_division(self):
        import warnings
        warnings.resetwarnings()
        warnings.simplefilter('ignore', RuntimeWarning)

        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(5.0))
        y = x1 / x0
        actual = y.data
        expected = np.array(np.inf)
        self.assertEqual(actual, expected)

        actual_grad0 = x0.grad
        self.assertTrue(actual_grad0 is None)

        actual_grad1 = x1.grad
        self.assertTrue(actual_grad1 is None)