import unittest
import numpy as np
from pyautodiff import Variable
from pyautodiff.analytic_function import *
from pyautodiff.calc_utils import numerical_diff, allclose


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        actual = y
        expected = Variable(np.array(4.0))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        actual = x.grad
        expected = Variable(np.array(6.0))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = exp(x)
        actual = y
        expected = Variable(np.array(np.exp(2.0)))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = exp(x)
        y.backward()
        actual = x.grad
        expected = Variable(np.array(np.exp(3.0)))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        num_grad = numerical_diff(exp, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class SinTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi/4.0))
        y = sin(x)
        actual = y
        expected =  Variable(np.array(np.sin(np.pi/4.0)))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(np.pi/4.0))
        y = sin(x)
        y.backward()
        actual = x.grad
        expected = Variable(np.array(np.cos(np.pi/4.0)))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = sin(x)
        y.backward()
        num_grad = numerical_diff(sin, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class CosTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi/4.0))
        y = cos(x)
        actual = y
        expected = Variable(np.array(np.cos(np.pi/4.0)))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(np.pi/4.0))
        y = cos(x)
        y.backward()
        actual = x.grad
        expected = Variable(np.array(-np.sin(np.pi/4.0)))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = cos(x)
        y.backward()
        num_grad = numerical_diff(cos, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class TanhTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(1.0))
        y = tanh(x)
        actual = y
        expected = Variable(np.array(np.tanh(1.0)))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(1.0))
        y = tanh(x)
        y.backward()
        actual = x.grad
        expected = 1 - y * y
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = tanh(x)
        y.backward()
        num_grad = numerical_diff(tanh, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)