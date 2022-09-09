import sys
sys.path.append("../.")


import unittest
import numpy as np
from autodiff.core_simple import Variable
from autodiff.functions import square
from autodiff.arithmetic_operator import add


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

    def test_multiple_use_of_same_variable(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()

        actual_val = y.data
        expected_val = np.array(6.0)
        self.assertEqual(actual_val, expected_val)
        
        actual_grad = x.grad
        expected_grad = np.array(2.0)
        self.assertEqual(actual_grad, expected_grad)


class ClearGradTest(unittest.TestCase):
    def test_cleargrad(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()

        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        actual = x.grad
        expected = np.array(3.0)
        self.assertEqual(actual, expected)



class GenerationTest(unittest.TestCase):
    def test_generation(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        
        actual_val = y.data
        expected_val = np.array(32.0)
        self.assertEqual(actual_val, expected_val)

        actual_grad = x.grad
        expected_grad = np.array(64.0)
        self.assertEqual(actual_grad, expected_grad)