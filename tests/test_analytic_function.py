import unittest
import numpy as np
from adfp.core import Variable
import adfp.functions.analytic_functions as F
from adfp.calc_utils import numerical_diff, allclose


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = F.square(x)
        actual = y
        expected = Variable(np.array(4.0))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = F.square(x)
        y.backward()
        actual = x.grad
        expected = Variable(np.array(6.0))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = F.square(x)
        y.backward()
        num_grad = numerical_diff(F.square, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = F.exp(x)
        actual = y
        expected = Variable(np.array(np.exp(2.0)))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = F.exp(x)
        y.backward()
        actual = x.grad
        expected = Variable(np.array(np.exp(3.0)))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = F.exp(x)
        y.backward()
        num_grad = numerical_diff(F.exp, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class SinTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi/4.0))
        y = F.sin(x)
        actual = y
        expected =  Variable(np.array(np.sin(np.pi/4.0)))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(np.pi/4.0))
        y = F.sin(x)
        y.backward()
        actual = x.grad
        expected = Variable(np.array(np.cos(np.pi/4.0)))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = F.sin(x)
        y.backward()
        num_grad = numerical_diff(F.sin, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class CosTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi/4.0))
        y = F.cos(x)
        actual = y
        expected = Variable(np.array(np.cos(np.pi/4.0)))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(np.pi/4.0))
        y = F.cos(x)
        y.backward()
        actual = x.grad
        expected = Variable(np.array(-np.sin(np.pi/4.0)))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = F.cos(x)
        y.backward()
        num_grad = numerical_diff(F.cos, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class TanhTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        actual = y
        expected = Variable(np.array(np.tanh(1.0)))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        y.backward()
        actual = x.grad
        expected = 1 - y * y
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = F.tanh(x)
        y.backward()
        num_grad = numerical_diff(F.tanh, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


class SigmoidTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = F.sigmoid(x)
        actual = y
        expected = Variable(np.array(1 / (1 + np.exp(-2.0))))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = F.sigmoid(x)
        y.backward()
        actual = x.grad
        z = np.exp(-2.0)
        expected = Variable(np.array(z / (1 + z) ** 2))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = F.sigmoid(x)
        y.backward()
        num_grad = numerical_diff(F.sigmoid, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)