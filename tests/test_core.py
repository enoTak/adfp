import unittest
import numpy as np

from adfp import Variable
from adfp.analytic_function import square, sin, exp
from adfp.module_config import use_simple_core


class CompositeTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))

        z = square(x) + square(y)
        actual = z
        expected = Variable(np.array(13.0))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = Variable(np.array(4.0))

        z = square(x) + square(y)
        z.backward()

        actual_x = x.grad
        expected_x = Variable(np.array(6.0))
        self.assertEqual(actual_x, expected_x)

        actual_y = y.grad
        expected_y = Variable(np.array(8.0))
        self.assertEqual(actual_y, expected_y)

    def test_multiple_use_of_same_variable(self):
        x = Variable(np.array(3.0))
        y = x + x
        y.backward()

        actual_val = y
        expected_val = Variable(np.array(6.0))
        self.assertEqual(actual_val, expected_val)
        
        actual_grad = x.grad
        expected_grad = Variable(np.array(2.0))
        self.assertEqual(actual_grad, expected_grad)


class ClearGradTest(unittest.TestCase):
    def test_cleargrad(self):
        x = Variable(np.array(3.0))
        y = x + x
        y.backward()

        x.cleargrad()
        y = (x + x) + x
        y.backward()
        actual = x.grad
        expected = Variable(np.array(3.0))
        self.assertEqual(actual, expected)


class GenerationTest(unittest.TestCase):
    def test_generation(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = square(a) + square(a)
        y.backward()
        
        actual_val = y
        expected_val = Variable(np.array(32.0))
        self.assertEqual(actual_val, expected_val)

        actual_grad = x.grad
        expected_grad = Variable(np.array(64.0))
        self.assertEqual(actual_grad, expected_grad)


class HigherOrderTest(unittest.TestCase):
    def test_higher_order(self):
        if use_simple_core:
            pass
        else:
            x = Variable(np.array(3.0))
            y = square(x)
            y.backward(create_graph=True)
            actual = x.grad
            expected = Variable(np.array(6.0))
            self.assertEqual(actual, expected)

            expected = [
                Variable(np.array(2.0)),
                Variable(np.array(0.0)),
                Variable(np.array(0.0)),
            ]
            for i,e in zip(range(2, 5), expected):
                gx = x.grad
                x.cleargrad()
                gx.backward(create_graph=True)
                a = x.grad
                self.assertEqual(a, e)