import unittest
import numpy as np
from adfp import Variable
from adfp .calc_utils import array_equal


class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 + x1
        actual = y
        expected = Variable(np.array(5.0))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 + x1
        y.backward()

        actual_0 = x0.grad
        expected_0 = Variable(np.array(1.0))
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = Variable(np.array(1.0))
        self.assertEqual(actual_1, expected_1)

    def test_combination_with_np_array(self):
        x0 = Variable(np.array(2.0))
        x1 = np.array(3.0)
        y = x0 + x1
        actual = y
        expected = Variable(np.array(5.0))
        self.assertEqual(actual, expected)

    def test_combination_with_np_array_with_list(self):
        x0 = np.array([3.0])
        x1 = Variable(np.array([2.0]))
        y = x0 + x1
        actual = y
        expected = Variable(np.array([5.0]))
        self.assertEqual(actual, expected)

    def test_combination_with_float(self):
        x0 = Variable(np.array(2.0))
        x1 = 3.0
        y = x0 + x1
        actual = y
        expected = Variable(np.array(5.0))
        self.assertEqual(actual, expected)

        x0 = 3.0
        x1 = Variable(np.array(2.0))
        y = x0 + x1
        actual = y.data
        expected = np.array(5.0)
        self.assertEqual(actual, expected)

    def test_broadcast(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        y.backward()

        actual = y
        expected = Variable(np.array([11, 12, 13]))
        self.assertTrue(array_equal(actual, expected))

        actual = x0.grad
        expected = Variable(np.array([1, 1, 1]))
        self.assertTrue(array_equal(actual, expected))

        actual = x1.grad
        expected = Variable(np.array([len(x0)]))
        self.assertTrue(array_equal(actual, expected))


class MulTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 * x1
        actual = y
        expected = Variable(np.array(6.0))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 * x1
        y.backward()

        actual_0 = x0.grad
        expected_0 = Variable(np.array(3.0))
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = Variable(np.array(2.0))
        self.assertEqual(actual_1, expected_1)

    def test_combination_with_np_array(self):
        x0 = Variable(np.array(2.0))
        x1 = np.array(3.0)
        y = x0 * x1
        actual = y
        expected = Variable(np.array(6.0))
        self.assertEqual(actual, expected)

    def test_combination_with_np_array_with_list(self):
        x0 = np.array([3.0])
        x1 = Variable(np.array([2.0]))
        y = x0 * x1
        actual = y
        expected = Variable(np.array([6.0]))
        self.assertEqual(actual, expected)

    def test_combination_with_float(self):
        x0 = Variable(np.array(2.0))
        x1 = 3.0
        y = x0 * x1
        actual = y
        expected = Variable(np.array(6.0))
        self.assertEqual(actual, expected)

        x0 = 3.0
        x1 = Variable(np.array(2.0))
        y = x0 * x1
        actual = y
        expected = Variable(np.array(6.0))
        self.assertEqual(actual, expected)

    def test_broadcast(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 * x1
        y.backward()

        actual = y
        expected = Variable(np.array([10, 20, 30]))
        self.assertTrue(array_equal(actual, expected))

        actual = x0.grad
        expected = Variable(np.array([10, 10, 10]))
        self.assertTrue(array_equal(actual, expected))

        actual = x1.grad
        expected = Variable(np.array([6]))
        self.assertTrue(array_equal(actual, expected))


class NegTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = -x
        actual = y
        expected = Variable(np.array(-2.0))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = -x
        y.backward()

        actual = x.grad
        expected = Variable(np.array(-1.0))
        self.assertEqual(actual, expected)


class SubTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(1.0))
        y = x0 - x1
        actual = y
        expected = Variable(np.array(1.0))
        self.assertEqual(actual, expected)

        x = Variable(np.array(2.0))
        y = x - 4.0
        actual = y
        expected = Variable(np.array(-2.0))
        self.assertEqual(actual, expected)

        y = 5.0 - x
        actual = y
        expected = Variable(np.array(3.0))
        self.assertEqual(actual, expected)


    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 - x1
        y.backward()

        actual_0 = x0.grad
        expected_0 = Variable(np.array(1.0))
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = Variable(np.array(-1.0))
        self.assertEqual(actual_1, expected_1)

    def test_broadcast(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 - x1
        y.backward()

        actual = y
        expected = Variable(np.array([-9, -8, -7]))
        self.assertTrue(array_equal(actual, expected))

        actual = x0.grad
        expected = Variable(np.array([1, 1, 1]))
        self.assertTrue(array_equal(actual, expected))

        actual = x1.grad
        expected = Variable(np.array([-3]))
        self.assertTrue(array_equal(actual, expected))


class DivTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(2.0))
        y = x0 / x1
        actual = y
        expected = Variable(np.array(1.0))
        self.assertEqual(actual, expected)
        
        x = Variable(np.array(2.0))
        y = x / 4.0
        actual = y
        expected = Variable(np.array(0.5))
        self.assertEqual(actual, expected)

        y = 5.0 / x
        actual = y
        expected = Variable(np.array(2.5))
        self.assertEqual(actual, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 - x1
        y.backward()

        actual_0 = x0.grad
        expected_0 = Variable(np.array(1.0))
        self.assertEqual(actual_0, expected_0)

        actual_1 = x1.grad
        expected_1 = Variable(np.array(-1.0))
        self.assertEqual(actual_1, expected_1)

    def test_zero_division(self):
        import warnings
        warnings.resetwarnings()
        warnings.simplefilter('ignore', RuntimeWarning)

        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(5.0))
        y = x1 / x0
        actual = y
        expected = Variable(np.array(np.inf))
        self.assertEqual(actual, expected)

        self.assertFalse(x0.is_updated_grad)
        self.assertFalse(x1.is_updated_grad)

    def test_broadcast(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 / x1
        y.backward()

        actual = y
        expected = Variable(np.array([0.1, 0.2, 0.3]))
        self.assertTrue(array_equal(actual, expected))

        actual = x0.grad
        expected = Variable(np.array([0.1, 0.1, 0.1]))
        self.assertTrue(array_equal(actual, expected))

        actual = x1.grad
        expected = Variable(np.array([-0.06]))
        self.assertTrue(array_equal(actual, expected))


class PowTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        c = 3.0
        y = x ** c
        actual = y
        expected = Variable(np.array(8.0))
        self.assertEqual(actual, expected)
        
    def test_backward(self):
        x = Variable(np.array(2.0))
        c = 3.0
        y = x ** c
        y.backward()

        actual = x.grad
        expected = Variable(np.array(12.0))
        self.assertEqual(actual, expected)