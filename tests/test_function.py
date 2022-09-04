import sys
sys.path.append("../.")


import unittest
import numpy as np
from numeric_ad.functions import *
from numeric_ad.binary_operators import *
from numeric_ad.core_simple import numerical_diff


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        actual = y.data
        expected = np.array(4.0)
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        actual = x.grad
        expected = np.array(6.0)
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = exp(x)
        actual = y.data
        expected = np.array(np.exp(2.0))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = exp(x)
        y.backward()
        actual = x.grad
        expected = np.array(np.exp(3.0))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        num_grad = numerical_diff(exp, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class CompositeTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))

        z = add(square(x), square(y))
        actual = z.data
        expected = np.array(13.0)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = Variable(np.array(4.0))

        z = add(square(x), square(y))
        z.backward()

        actual_x = x.grad
        expected_x = np.array(6.0)
        self.assertEqual(actual_x, expected_x)

        actual_y = y.grad
        expected_y = np.array(8.0)
        self.assertEqual(actual_y, expected_y)
